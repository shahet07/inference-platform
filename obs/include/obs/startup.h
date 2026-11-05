#pragma once

#include <locale>

namespace obs {

// Call at the start of main, before any thread. libstdc++ fills its locale's character caches
// lazily and without a lock, the first time each character is used, and regex compilation (which
// the HTTP library does on many threads) uses them. ThreadSanitizer reports that as a data race.
// Filling them all up front means every later use only reads.
inline void primeLocaleCaches() {
  const auto& facet = std::use_facet<std::ctype<char>>(std::locale());
  for (int c = -128; c < 128; ++c) {
    (void)facet.narrow(static_cast<char>(c), '\0');
    (void)facet.widen(static_cast<char>(c));
  }
}

}  // namespace obs
