#include "matmul.h"

#include <vector>

namespace mm {

namespace {
std::vector<Variant>& registry() {
  static std::vector<Variant> variants;
  return variants;
}
}  // namespace

void register_variant(const Variant& v) { registry().push_back(v); }

int variant_count() { return static_cast<int>(registry().size()); }

const Variant& variant_at(int i) { return registry()[i]; }

}  // namespace mm
