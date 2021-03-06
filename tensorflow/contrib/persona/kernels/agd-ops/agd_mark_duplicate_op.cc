/* Copyright 2019 École Polytechnique Fédérale de Lausanne. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
#include <boost/functional/hash.hpp>
#include <google/dense_hash_map>

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        data->release();
      }
   }


  using namespace std;
  using namespace errors;
  using namespace format;

  inline bool operator==(const Position& lhs, const Position& rhs) {
    return (lhs.ref_index() == rhs.ref_index() && lhs.position() == rhs.position());
  }

  class AGDMarkDuplicatesOp : public OpKernel {
  public:
    AGDMarkDuplicatesOp(OpKernelConstruction *context) : OpKernel(context) {
      signature_map_ = new SignatureMap();
      Signature sig;
      signature_map_->set_empty_key(sig);
    }

    ~AGDMarkDuplicatesOp() {
      LOG(INFO) << "Found a total of " << num_dups_found_ << " duplicates.";
      core::ScopedUnref unref_listpool(bufferpair_pool_);
      delete signature_map_;
    }

    Status GetOutputBufferPair(OpKernelContext* ctx, ResourceContainer<BufferPair> **ctr)
    {
      TF_RETURN_IF_ERROR(bufferpair_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("marked_results", ctx));
      return Status::OK();
    }
    
    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufferpair_pool_));

      return Status::OK();
    }

    inline int parseNextOp(const char *ptr, char &op, int &num)
    {
      num = 0;
      const char * begin = ptr;
      for (char curChar = ptr[0]; curChar != 0; curChar = (++ptr)[0])
      {
        int digit = curChar - '0';
        if (digit >= 0 && digit <= 9) num = num*10 + digit;
        else break;
      }
      op = (ptr++)[0];
      return ptr - begin;
    }
   
    Status CalculatePosition(const Alignment *result,
        uint32_t &position) {
      // figure out the 5' position
      // the result->location is already genome relative, we shouldn't have to worry 
      // about negative index after clipping, but double check anyway
      // cigar parsing adapted from samblaster
      const char* cigar;
      size_t cigar_len;
      cigar = result->cigar().c_str();
      cigar_len = result->cigar().length();
      int ralen = 0, qalen = 0, sclip = 0, eclip = 0;
      bool first = true;
      char op;
      int op_len;
      while(cigar_len > 0) {
        size_t len = parseNextOp(cigar, op, op_len);
        cigar += len;
        cigar_len -= len;         
        //LOG(INFO) << "cigar was " << op_len << " " << op;
        //LOG(INFO) << "cigar len is now: " << cigar_len;
        if (op == 'M' || op == '=' || op == 'X')
        {
          ralen += op_len;
          qalen += op_len;
          first = false;
        }
        else if (op == 'S' || op == 'H')
        {
          if (first) sclip += op_len;
          else       eclip += op_len;
        }
        else if (op == 'D' || op == 'N')
        {
          ralen += op_len;
        }
        else if (op == 'I')
        {
          qalen += op_len;
        }
        else
        {
          return Internal("Unknown opcode ", string(&op, 1), " in CIGAR string: ", string(cigar, cigar_len));
        }
      }
      //LOG(INFO) << "the location is: " << result->location_;
      if (IsForwardStrand(result->flag())) {
        position = static_cast<uint32_t>(result->position().position() - sclip);
      } else {
        // im not 100% sure this is correct ...
        // but if it goes for every signature then it shouldn't matter
        position = static_cast<uint32_t>(result->position().position() + ralen + eclip - 1);
      }
      //LOG(INFO) << "position is now: " << position;
      if (position < 0)
        return Internal("A position after applying clipping was < 0! --> ", position);
      return Status::OK();
    }

    Status MarkDuplicate(const Alignment* result, AlignmentResultBuilder &builder) {
      Alignment result_out;
      result_out.CopyFrom(*result); // simple copy suffices
      result_out.set_flag(result_out.flag()| ResultFlag::PCR_DUPLICATE);
      // yes we are copying and rebuilding the entire structure
      // modifying in place is a huge pain in the ass, and the results aren't that
      // big anyway
      builder.AppendAlignmentResult(result_out);
      return Status::OK();
    }

    Status ProcessOrphan(const Alignment* result, AlignmentResultBuilder &builder) {
      Signature sig;
      sig.is_forward = IsForwardStrand(result->flag());
      TF_RETURN_IF_ERROR(CalculatePosition(result, sig.position));

      //LOG(INFO) << "sig is: " << sig.ToString();
      // attempt to find the signature
      auto sig_map_iter = signature_map_->find(sig);
      if (sig_map_iter == signature_map_->end()) { // not found, insert it
        signature_map_->insert(make_pair(sig, 1));
        // its the first here, others will be marked dup
        builder.AppendAlignmentResult(*result);
        return Status::OK();
      } else { 
        // found, mark a dup
        num_dups_found_++;
        return MarkDuplicate(result, builder);
      }
    }

    void Compute(OpKernelContext* ctx) override {
      if (!bufferpair_pool_) {
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
      }

      LOG(INFO) << "Starting duplicate mark";
      const Tensor* results_t, *num_results_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_results_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      auto results_handle = results_t->vec<string>();
      auto num_results = num_results_t->scalar<int32>()();
      auto rmgr = ctx->resource_manager();
      ResourceContainer<Data> *results_container;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results_handle(0), results_handle(1), &results_container));
      AGDResultReader results_reader(results_container, num_results);

      // get output buffer pairs (pair holds [index, data] to construct
      // the results builder for output
      ResourceContainer<BufferPair> *output_bufferpair_container;
      OP_REQUIRES_OK(ctx, GetOutputBufferPair(ctx, &output_bufferpair_container));
      auto output_bufferpair = output_bufferpair_container->get();
      AlignmentResultBuilder results_builder;
      results_builder.SetBufferPair(output_bufferpair);


      Alignment result;
      Alignment mate;
      Status s = results_reader.GetNextResult(result);

      // this detection logic adapted from SamBlaster
      while (s.ok()) {
        if (!IsPrimary(result.flag()))
          OP_REQUIRES_OK(ctx, Internal("Non-primary result detected in primary result column at location ",
                result.position().DebugString()));
        if (!IsPaired(result.flag())) {
          // we have a single alignment
          if (IsUnmapped(result.flag())) {
            s = results_reader.GetNextResult(result);
            continue;
          }

          //LOG(INFO) << "processing mapped orphan at " << result->location_;
          OP_REQUIRES_OK(ctx, ProcessOrphan(&result, results_builder));

        } else { // we have a pair, get the mate
          OP_REQUIRES_OK(ctx, results_reader.GetNextResult(mate));

          OP_REQUIRES(ctx, (result.next_position() == mate.position()) && (mate.next_position() == result.position()),
              Internal("Malformed pair or the data is not in metadata (QNAME) order. At index: ", results_reader.GetCurrentIndex()-1,
                "result 1: ", result.DebugString(), " result2: ", mate.DebugString()));

          //LOG(INFO) << "processing mapped pair at " << result->location_ << ", " << mate->location_;

          if (IsUnmapped(result.flag()) && IsUnmapped(mate.flag())) {
            s = results_reader.GetNextResult(result);
            continue;
          }
          
          if (IsUnmapped(result.flag()) && IsMapped(mate.flag())) { // treat as single
            OP_REQUIRES_OK(ctx, ProcessOrphan(&mate, results_builder));
          } else if (IsUnmapped(mate.flag()) && IsMapped(result.flag())) {
            OP_REQUIRES_OK(ctx, ProcessOrphan(&result, results_builder));
          } else {
            Signature sig;
            sig.is_forward = IsForwardStrand(result.flag());
            sig.is_mate_forward = IsForwardStrand(mate.flag());
            // adjust position depending on cigar clipping
            OP_REQUIRES_OK(ctx, CalculatePosition(&result, sig.position));
            OP_REQUIRES_OK(ctx, CalculatePosition(&mate, sig.position_mate));

            // attempt to find the signature
            auto sig_map_iter = signature_map_->find(sig);
            if (sig_map_iter == signature_map_->end()) { // not found, insert it
              signature_map_->insert(make_pair(sig, 1));
              results_builder.AppendAlignmentResult(result);
              results_builder.AppendAlignmentResult(mate);
            } else { 
              // found, mark a dup
              LOG(INFO) << "omg we found a duplicate";
              OP_REQUIRES_OK(ctx, MarkDuplicate(&result, results_builder));
              OP_REQUIRES_OK(ctx, MarkDuplicate(&mate, results_builder));
              num_dups_found_++;
            }
          }
        }
        s = results_reader.GetNextResult(result);
      } // while s is ok()

      // done
      resource_releaser(results_container);
      LOG(INFO) << "DONE running mark duplicates!! Found so far: " << num_dups_found_;

    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;

    struct Signature {
      uint32_t position = 0;
      uint32_t ref_index = 0;
      uint32_t position_mate = 0;
      uint32_t ref_index_mate = 0;
      bool is_forward = true;
      bool is_mate_forward = true;
      bool operator==(const Signature& s) {
        return (s.position == position) && (s.position_mate == position_mate)
          && (s.is_forward == is_forward) && (s.is_mate_forward == is_mate_forward)
                && (s.ref_index == ref_index) && (s.ref_index_mate == ref_index_mate);
      }
      string ToString() const {
        return string("pos: ") + to_string(position) + " matepos: " + to_string(position_mate)
          + " isfor: " + to_string(is_forward) + " ismatefor: " + to_string(is_mate_forward) ;
      }
    };

    struct EqSignature {
      bool operator()(Signature sig1, Signature sig2) const {
        return (sig1.position == sig2.position) && (sig1.position_mate == sig2.position_mate) 
          && (sig1.is_forward == sig2.is_forward) && (sig1.is_mate_forward == sig2.is_mate_forward)
          && (sig1.ref_index == sig2.ref_index) && (sig1.ref_index_mate == sig2.ref_index_mate);
      }
    };

    struct SigHash {
      size_t operator()(Signature const& s) const {
        size_t p = hash<uint32_t>{}(s.position);
        size_t pm = hash<uint32_t>{}(s.position_mate);
        size_t ri = hash<uint32_t>{}(s.ref_index);
        size_t rim = hash<uint32_t>{}(s.ref_index_mate);
        size_t i = hash<bool>{}(s.is_forward);
        size_t m = hash<bool>{}(s.is_mate_forward);
        // maybe this is too expensive
        boost::hash_combine(ri, rim);
        boost::hash_combine(p, pm);
        boost::hash_combine(i, m);
        boost::hash_combine(p, i);
        boost::hash_combine(p, ri);
        //LOG(INFO) << "hash was called on " << s.ToString() << " and value was: " << p;
        return p;
      }
    };

    typedef google::dense_hash_map<Signature, int, SigHash, EqSignature> SignatureMap;
    SignatureMap* signature_map_;

    int num_dups_found_ = 0;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDMarkDuplicates").Device(DEVICE_CPU), AGDMarkDuplicatesOp);
} //  namespace tensorflow {
