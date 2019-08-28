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
#include "tensorflow/contrib/persona/kernels/snap-align/single_executor.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  SnapSingleExecutor::SnapSingleExecutor(Env *env, GenomeIndex *index, AlignerOptions *options,
                                         int num_threads, int capacity) : index_(index),
                                                                          options_(options),
                                                                          num_threads_(num_threads),
                                                                          capacity_(capacity) {
    genome_ = index_->getGenome();
    // create a threadpool to execute stuff
    workers_.reset(new thread::ThreadPool(env, "SnapSingle", num_threads_));
    request_queue_.reset(new ConcurrentQueue<std::shared_ptr<ResourceContainer<ReadResource>>>(capacity));
    auto s = snap_wrapper::init();
    if (s.ok()) {
      init_workers();
    } else {
      LOG(ERROR) << "Unable to run snap_wrapper::init()";
      compute_status_ = s;
    }
  }

  SnapSingleExecutor::~SnapSingleExecutor() {
    if (!run_) {
      LOG(ERROR) << "Unable to safely wait in ~SnapAlignSingleOp for all threads. run_ was toggled to false\n";
    }
    run_ = false;
    request_queue_->unblock();
    while (num_active_threads_.load(std::memory_order_relaxed) > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  Status SnapSingleExecutor::EnqueueChunk(std::shared_ptr<ResourceContainer<ReadResource> > chunk) {
    if (!compute_status_.ok()) return compute_status_;
    if (!request_queue_->push(chunk))
      return Internal("Single executor failed to push to request queue");
    else
      return Status::OK();
  }

  void SnapSingleExecutor::init_workers() {

    auto aligner_func = [this]() {
      //std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
      int my_id = id_.fetch_add(1, memory_order_relaxed);

      int capacity = request_queue_->capacity();

      unsigned alignmentResultBufferCount;
      if (options_->maxSecondaryAlignmentAdditionalEditDistance < 0) {
        alignmentResultBufferCount = 1; // For the primary alignment
      } else {
        alignmentResultBufferCount =
                BaseAligner::getMaxSecondaryResults(options_->numSeedsFromCommandLine, options_->seedCoverage,
                                                    MAX_READ_LENGTH, options_->maxHits, index_->getSeedLength()) +
                1; // +1 for the primary alignment
      }
      size_t alignmentResultBufferSize =
              sizeof(SingleAlignmentResult) * (alignmentResultBufferCount + 1); // +1 is for primary result

      unique_ptr<BigAllocator> allocator(new BigAllocator(BaseAligner::getBigAllocatorReservation(index_, true,
                                                                                                  options_->maxHits,
                                                                                                  MAX_READ_LENGTH,
                                                                                                  index_->getSeedLength(),
                                                                                                  options_->numSeedsFromCommandLine,
                                                                                                  options_->seedCoverage,
                                                                                                  options_->maxSecondaryAlignmentsPerContig)
                                                          + alignmentResultBufferSize));

      /*LOG(INFO) << "reservation: " << BaseAligner::getBigAllocatorReservation(index, true,
            options->maxHits, MAX_READ_LENGTH, index->getSeedLength(), options->numSeedsFromCommandLine, options->seedCoverage, options->maxSecondaryAlignmentsPerContig)
          + alignmentResultBufferSize;*/

      BaseAligner *base_aligner = new(allocator.get()) BaseAligner(
              index_,
              options_->maxHits,
              options_->maxDist,
              MAX_READ_LENGTH,
              options_->numSeedsFromCommandLine,
              options_->seedCoverage,
              options_->minWeightToCheck,
              options_->extraSearchDepth,
              false, false, false, // stuff that would decrease performance without impacting quality
              options_->maxSecondaryAlignmentsPerContig,
              nullptr, nullptr, // Uncached Landau-Vishkin
              nullptr, // No need for stats
              allocator.get()
      );

      allocator->checkCanaries();

      base_aligner->setExplorePopularSeeds(options_->explorePopularSeeds);
      base_aligner->setStopOnFirstHit(options_->stopOnFirstHit);

      SingleAlignmentResult primaryResult;
      vector<SingleAlignmentResult> secondaryResults;
      secondaryResults.resize(alignmentResultBufferCount);

      int num_secondary_results;
      vector<AlignmentResultBuilder> result_builders;
      string cigarString;
      Read snap_read;
      LandauVishkinWithCigar lvc;
      size_t num_columns;

      vector<BufferPair *> result_bufs;
      ReadResource *subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;

      while (run_) {
        // reads must be in this scope for the custom releaser to work!
        shared_ptr<ResourceContainer < ReadResource> > reads_container;
        if (!request_queue_->peek(reads_container)) {
          continue;
        }
        ScopeDropIfEqual<decltype(reads_container)> scope_dropper(*request_queue_, reads_container);

        auto *reads = reads_container->get();

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        while (io_chunk_status.ok()) {
          num_columns = result_bufs.size();

          if (num_columns > result_builders.size()) {
            result_builders.resize(num_columns);
          }

          for (int i = 0; i < num_columns; i++) {
            result_builders[i].SetBufferPair(result_bufs[i]);
          }

          for (subchunk_status = subchunk_resource->get_next_record(snap_read); subchunk_status.ok();
               subchunk_status = subchunk_resource->get_next_record(snap_read)) {
            cigarString.clear();
            snap_read.clip(options_->clipping);
            auto below_min_length = snap_read.getDataLength() < options_->minReadLength;
            auto too_many_ns = snap_read.countOfNs() > options_->maxDist;
            if (below_min_length or too_many_ns) {
#ifdef ENABLE_TRACING
              if (below_min_length) {
                num_dropped_reads_.fetch_add(1, memory_order_relaxed);
              } else {
                too_many_Ns_.fetch_add(1, memory_order_relaxed);
              }
#endif
              primaryResult.status = AlignmentResult::NotFound;
              primaryResult.location = InvalidGenomeLocation;
              primaryResult.mapq = 0;
              primaryResult.direction = FORWARD;
              auto s = snap_wrapper::WriteSingleResult(snap_read, primaryResult, result_builders[0], genome_, &lvc,
                                                       false, options_->useM);

              if (!s.ok()) {
                LOG(ERROR) << "adjustResults did not return OK!!!";
              }
              for (decltype(num_columns) i = 1; i < num_columns; i++) {
                // fill the columns with empties to maintain index equivalence
                result_builders[i].AppendEmpty();
              }
              continue;
            }
#ifdef ENABLE_TRACING
            good_reads_.fetch_add(1, memory_order_relaxed);
#endif

            base_aligner->AlignRead(
                    &snap_read,
                    &primaryResult,
                    options_->maxSecondaryAlignmentAdditionalEditDistance,
                    alignmentResultBufferCount,
                    &num_secondary_results,
                    num_columns-1, // maximum number of secondary results
                    &secondaryResults[0] //secondaryResults
            );

            // First, write the primary results
            auto s = snap_wrapper::WriteSingleResult(snap_read, primaryResult, result_builders[0], genome_, &lvc,
                                                     false, options_->useM);
#ifdef ENABLE_TRACING
            if (primaryResult.status == AlignmentResult::NotFound) {
                not_found_.fetch_add(1, memory_order_relaxed);
            } else if (primaryResult.location == InvalidGenomeLocation) {
                unknown_location_.fetch_add(1, memory_order_relaxed);
            }
#endif

            if (!s.ok()) {
              LOG(ERROR) << "adjustResults did not return OK!!!";
              compute_status_ = s;
              break;
            }

            // Then write the secondary results if we specified them
            for (decltype(num_secondary_results) i = 0; i < num_secondary_results; i++) {
              s = snap_wrapper::WriteSingleResult(snap_read, secondaryResults[i], result_builders[i + 1], genome_,
                                                  &lvc, true, options_->useM);
              if (!s.ok()) {
                LOG(ERROR) << "adjustResults did not return OK!!!";
                break;
              }
            }

            if (s.ok()) {
              for (decltype(num_secondary_results) i = num_secondary_results+1; i < num_columns; i++) {
                // fill the columns with empties to maintain index equivalence
                result_builders[i].AppendEmpty();
              }
            } else {
              compute_status_ = s;
              break;
            }
          }

          if (!IsResourceExhausted(subchunk_status)) {
            LOG(ERROR) << "Subchunk iteration ended without resource exhaustion!";
            compute_status_ = subchunk_status;
            break;
          } else if (!compute_status_.ok()) {
            break;
          }

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        }

        auto compute_error = !compute_status_.ok();
        if (!IsResourceExhausted(io_chunk_status) || compute_error) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status
                     << "\n";
          if (!compute_error)
            compute_status_ = io_chunk_status;
          run_ = false;
          break;
        }
      }

      base_aligner->~BaseAligner(); // This calls the destructor without calling operator delete, allocator owns the memory.
      VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_.fetch_sub(1, memory_order_relaxed);
    };
    num_active_threads_ = num_threads_;
    for (int i = 0; i < num_threads_; i++)
      workers_->Schedule(aligner_func);

#ifdef ENABLE_TRACING
    worker_thread_ = std::thread([this]() {
      while (run_) {
        this_thread::sleep_for(chrono::seconds(1));
        auto dropped_reads_in_interval = num_dropped_reads_.exchange(0, memory_order_relaxed);
        auto many_ns_in_interval = too_many_Ns_.exchange(0, memory_order_relaxed);
        auto good_reads_in_interval = good_reads_.exchange(0, memory_order_relaxed);
        auto not_found_in_interval = not_found_.exchange(0, memory_order_relaxed);
        auto unknown_location_in_interval = unknown_location_.exchange(0, memory_order_relaxed);
        //auto total_events = dropped_reads_in_interval + many_ns_in_interval;

        /*
        if (total_events > 1000) {
          LOG(WARNING) << "High number of read issues. Too short: " << dropped_reads_in_interval << ", high Ns: " << many_ns_in_interval;
          if (dropped_reads_in_interval > 10000) {
            run_ = false;
            LOG(ERROR) << "Too many dropped reads in interval. Shutting down";
          }
        }
         */
        LOG(INFO) << ">>>" << chrono::system_clock::now().time_since_epoch().count() << "," << good_reads_in_interval << "," << not_found_in_interval << "," << unknown_location_in_interval << "," << dropped_reads_in_interval << "," << many_ns_in_interval;

        //auto reads_in_interval = num_reads_.exchange(0, memory_order_relaxed);
        //LOG(INFO) << ">>" << chrono::system_clock::now().time_since_epoch().count() << "," << reads_in_interval;
      }
    });
#endif
  }

  Status SnapSingleExecutor::ok() const {
    return compute_status_;
  }
} // namespace tensorflow {
