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
#include <cstddef>
#include <memory>
#include "data.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

    class Buffer : public Data {
    private:
      std::unique_ptr<char[]> buf_ = nullptr;
      std::size_t size_ = 0, allocation_ = 0, extend_extra_ = 0;

      void resize_internal(size_t total_size);

    public:

      ~Buffer() override = default;
      TF_DISALLOW_COPY_AND_ASSIGN(Buffer);
      Buffer(Buffer&&) = default;
      Buffer & operator=(Buffer&&) = default;

      Buffer(decltype(size_) initial_size = 2 * 1024 * 1024,
             decltype(size_) extend_extra = 8 * 1024 * 1024);

        bool WriteBuffer(const char *content, std::size_t content_size);
        bool AppendBuffer(const char *content, std::size_t content_size);

        // ensures that the total capacity is at least `capacity`
        size_t reserve(decltype(allocation_) capacity, bool walk_allocation=false);

        // resizes the actual size to to total_size
        // returns an error if total_size > allocation_
        // should call `reserve` first to be safe
        Status resize(decltype(size_) total_size) override;

        // extends the current allocation by `extend_size`
        // does not affect size_
        void extend_allocation(decltype(size_) extend_size);

        // extends the current size by extend_size
        // returns an error if size_+extend_size > allocation
        // should call `extend_allocation` first to be safe
        void extend_size(decltype(size_) extend_size);

        char& operator[](std::size_t idx) const;

        void reset();
        virtual const char* data() const override;

        void release() override;

        virtual std::size_t size() const override;
        virtual char* mutable_data() override;
        std::size_t capacity() const;
    };
} // namespace tensorflow {
