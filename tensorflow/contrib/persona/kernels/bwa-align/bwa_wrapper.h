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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include <vector>
#include <memory>
#include <string>
#include <array>
#include "bwa/bwamem.h"
#include "bwa/bwa.h"
#include "bwa/bwt.h"

namespace bwa_wrapper {
    using namespace tensorflow;
    using namespace std;

    class BWAAligner
    {
    public:
      BWAAligner(const mem_opt_t *options, const bwaidx_t *index_resource, size_t max_read_len) :
        index_(index_resource), options_(options), max_read_len_(max_read_len) {
   
          seq = new char[max_read_len];
          seqmate = new char[max_read_len];
          aux_ = smem_aux_init();
      }

      ~BWAAligner() {
        delete [] seq;
        delete [] seqmate;
        smem_aux_destroy(aux_);
      }

      Status AlignSubchunkSingle(ReadResource* subchunk, vector<AlignmentResultBuilder>& result_builder);
      // align a whole subchunk since BWA infers insert distance from the data
      Status AlignSubchunk(ReadResource *subchunk, size_t index, vector<mem_alnreg_v>& regs);
      
      Status FinalizeSubchunk(ReadResource *subchunk, size_t regs_index, vector<mem_alnreg_v>& regs, 
          mem_pestat_t pes[4], vector<AlignmentResultBuilder>& result_builders);

    private:
      typedef struct {
        bwtintv_v mem, mem1, *tmpv[2];
      } smem_aux_t;
      
      // duplicated from BWA to avoid modifying it more
      // this isnt ideal but it should work ...

      static smem_aux_t *smem_aux_init()
      {
        smem_aux_t *a;
        a = (smem_aux_t*)calloc(1, sizeof(smem_aux_t));
        a->tmpv[0] = (bwtintv_v*)calloc(1, sizeof(bwtintv_v));
        a->tmpv[1] = (bwtintv_v*)calloc(1, sizeof(bwtintv_v));
        return a;
      }
        
      static void smem_aux_destroy(smem_aux_t *a)
      {	
        free(a->tmpv[0]->a); free(a->tmpv[0]);
        free(a->tmpv[1]->a); free(a->tmpv[1]);
        free(a->mem.a); free(a->mem1.a);
        free(a);
      }

      // we dont own these
      const mem_opt_t *options_;
      const bwaidx_t *index_;
      size_t max_read_len_;
      char * seq;
      char * seqmate;
	
      smem_aux_t *aux_;
      
      void ProcessResult(mem_aln_t* bwaresult, mem_aln_t* bwamate, Alignment& result, string& cigar);

      const string placeholder = "i'm mr meeseeks look at me!";

    };
  
}
