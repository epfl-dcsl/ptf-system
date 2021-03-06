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
#pragma once

#include "tensorflow/contrib/persona/kernels/agd-format/agd_reads.h"
#include "tensorflow/core/lib/core/errors.h"
#include <atomic>
#include <vector>
#include <memory>
#include "bwa/bwamem.h"
#include "bwa/bntseq.h"

namespace tensorflow {

  class BWAReadResource : public AGDReadResource {

    public:
      explicit BWAReadResource() = default;
      explicit BWAReadResource(std::size_t num_records, DataContainer *bases, DataContainer *quals, DataContainer *meta) :
        AGDReadResource(num_records, bases, quals, meta) {
        regs_.resize(num_records);
      }
      explicit BWAReadResource(std::size_t num_records, DataContainer *bases, DataContainer *quals) : 
        AGDReadResource(num_records, bases, quals) {
        regs_.resize(num_records);
      }

      Status split(std::size_t chunk, std::vector<BufferList*>& bl) override {
        TF_RETURN_IF_ERROR(AGDReadResource::split(chunk, bl));
        auto total = num_records();
        intervals_.clear();
        for (size_t i = 0; i < total; i += chunk)
          intervals_.push_back(i);
   
        if (intervals_.size() != sub_resources_.size())
          return errors::Internal("BWA read resource got different number intervals and rsrcs??");

        outstanding_subchunks_.store(sub_resources_.size(), std::memory_order_relaxed);

        return Status::OK();
      }
    
      Status get_next_subchunk(ReadResource **rr, size_t* interval) {
        auto a = sub_resource_index_.fetch_add(1, std::memory_order_relaxed);
        if (a >= sub_resources_.size()) {
          return errors::ResourceExhausted("No more BWA/AGD subchunks");
        } 
        
        *rr = &sub_resources_[a];
        *interval = intervals_[a];
        return Status::OK();
      }
    
      Status get_next_subchunk(ReadResource **rr, std::vector<BufferPair*>& b, size_t* interval) {
        auto a = sub_resource_index_.fetch_add(1, std::memory_order_relaxed);
        if (a >= sub_resources_.size()) {
          return errors::ResourceExhausted("No more BWA/AGD subchunks");
        }

        *rr = &sub_resources_[a];
        b.clear();
        for (auto bl : buffer_lists_) {
          b.push_back(&(*bl)[a]);
        }
        *interval = intervals_[a];
        return Status::OK();
      }

      void wait_for_ready() const {
        if (outstanding_subchunks_.load(std::memory_order_relaxed) != 0) {
          mutex_lock l(mu_);
          while (outstanding_subchunks_.load(std::memory_order_relaxed) == 0) {
            ready_cv_.wait(l);
          }
        }
      }
      
      void decrement_outstanding() {
        auto previous = outstanding_subchunks_.fetch_sub(1, std::memory_order_relaxed);
        if (previous == 1) {
          ready_cv_.notify_one();
        }
      }
  
      void reset() {
        intervals_.clear();
        outstanding_subchunks_.store(0, std::memory_order_relaxed);
      }

      void reset_subchunks() {
        sub_resource_index_.store(0, std::memory_order_relaxed);
      }
      
      std::vector<mem_alnreg_v>& get_regs() { return regs_; }
      mem_pestat_t* get_pes() { return pes; }

    private:
      std::vector<mem_alnreg_v> regs_;
      mem_pestat_t pes[4];
      std::vector<size_t> intervals_;
      mutable std::atomic_size_t outstanding_subchunks_;;
      mutable mutex mu_;
      mutable condition_variable ready_cv_;
  };


}
