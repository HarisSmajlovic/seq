#pragma once

#include "dsl/dsl.h"

namespace seq {

class Codim : public DSL {
public:
  std::string getName() const override { return "Codim"; }
  void addIRPasses(ir::transform::PassManager *pm, bool debug) override;
};

} // namespace seq
