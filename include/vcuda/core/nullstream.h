// SPDX-License-Identifier: MIT
#ifndef VCUDA_CORE_NULLSTREAM_H
#define VCUDA_CORE_NULLSTREAM_H 1
#include <ostream>
#include <streambuf>

namespace vcuda {
  namespace core {
    /*------------------------------------------------------------------------*/
    /*! Disable logging -- from https://stackoverflow.com/a/11826666. */
    /*------------------------------------------------------------------------*/
    class NullStream : public std::ostream {
      private:
        class NullBuffer : public std::streambuf {
          public:
            int overflow(int c) { return c; }
        };

        NullBuffer m_sb;

      public:
        NullStream() : std::ostream(&m_sb) {}
    };
  }
}

#endif // VCUDA_CORE_NULLSTREAM_H
