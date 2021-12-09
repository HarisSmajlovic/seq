#include "codim.h"
#include "arithmetics.h"

namespace seq {

void Codim::addIRPasses(ir::transform::PassManager *pm, bool debug) {
  pm->registerPass(std::make_unique<ArithmeticsOptimizations>());
}

} // namespace seq
