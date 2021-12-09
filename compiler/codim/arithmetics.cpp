#include "arithmetics.h"
#include "sir/util/cloning.h"
#include "sir/util/irtools.h"
#include "sir/util/matching.h"
#include <iterator>
#include <math.h>

namespace seq {

using namespace ir;

/*
 * Binary expression tree
 */
const std::string codimModule = "std.codim";

const int BET_ADD_OP = 1;
const int BET_MUL_OP = 2;
const int BET_POW_OP = 3;
const int BET_REVEAL_OP = 4;
const int BET_MATMUL_OP = 5;
const int BET_NORM_OP = 6;
const int BET_OTHER_OP = 7;

class BETNode {
  int64_t value;
  int variableId;
  seq::ir::Var *var;
  int op;
  BETNode *leftChild;
  BETNode *rightChild;
  bool expanded;
  bool constant;

  std::vector<seq::ir::Var *> vars;

public:
  BETNode();
  BETNode(int variableId);
  BETNode(int variableId, int op, bool expanded, int64_t value, bool constant);
  BETNode(int variableId, int op, BETNode *leftChild, BETNode *rightChild,
          bool expanded, int64_t value, bool constant);
  ~BETNode() {
    if (leftChild)
      delete leftChild;
    if (rightChild)
      delete rightChild;
  }

  void setVariableId(int variableId) { this->variableId = variableId; }
  void setVar(seq::ir::Var *var) { this->var = var; }
  void setOperator(int op) { this->op = op; }
  void setLeftChild(BETNode *leftChild) { this->leftChild = leftChild; }
  void setRightChild(BETNode *rightChild) { this->rightChild = rightChild; }
  void setExpanded() { expanded = true; }
  int getVariableId() { return variableId; }
  seq::ir::Var *getVar() {return var; }
  std::vector<seq::ir::Var *> getVars() {return vars; }
  int getOperator() { return op; }
  int64_t getValue() { return value; }
  BETNode *getLeftChild() { return leftChild; }
  BETNode *getRightChild() { return rightChild; }
  bool isExpanded() { return expanded; }
  bool isLeaf() { return !leftChild && !rightChild; }
  bool isAdd() { return op == BET_ADD_OP; }
  bool isMul() { return op == BET_MUL_OP; }
  bool isPow() { return op == BET_POW_OP; }
  bool isConstant() { return constant; }
  void replace(BETNode *);
  BETNode *copy();
  void print();
  void parseVars();
  std::vector<seq::ir::Value *> matchOptFuncArgs(seq::ir::Module *M);
};

BETNode::BETNode()
    : value(1), variableId(0), op(0), leftChild(nullptr), rightChild(nullptr),
      expanded(false), constant(false) {}

BETNode::BETNode(int variableId)
    : value(1), variableId(variableId), op(0), leftChild(nullptr), rightChild(nullptr),
      expanded(false), constant(false) {}

BETNode::BETNode(int variableId, int op, bool expanded, int64_t value, bool constant)
    : value(value), variableId(variableId), op(op), leftChild(nullptr),
      rightChild(nullptr), expanded(expanded), constant(constant) {}

BETNode::BETNode(int variableId, int op, BETNode *leftChild, BETNode *rightChild,
                 bool expanded, int64_t value, bool constant)
    : value(value), variableId(variableId), op(op), leftChild(leftChild),
      rightChild(rightChild), expanded(expanded), constant(constant) {}

void BETNode::replace(BETNode *other) {
  op = other->getOperator();
  leftChild = other->getLeftChild();
  rightChild = other->getRightChild();
  expanded = other->isExpanded();
  value = other->getValue();
  constant = other->isConstant();
}

BETNode *BETNode::copy() {
  auto *newNode = new BETNode(variableId, op, expanded, value, constant);
  auto *lc = getLeftChild();
  auto *rc = getRightChild();
  if (lc)
    newNode->setLeftChild(lc->copy());
  if (rc)
    newNode->setRightChild(rc->copy());
  return newNode;
}

void BETNode::parseVars() {
  vars.empty();
  
  if (isConstant())
    return;
  if (isLeaf()) {
    auto *v = getVar();
    if (v->getName().find("__var__") != std::string::npos) return;
    vars.push_back(v);
    return;
  }

  getLeftChild()->parseVars();
  for (auto *v : getLeftChild()->vars) vars.push_back(v);
  getRightChild()->parseVars();
  for (auto *v : getRightChild()->vars) vars.push_back(v);
}

void BETNode::print() {
  std::cout << op << " " << variableId
            << (constant ? " Is constant " : " Not constant ") << value << std::endl;
  if (leftChild)
    leftChild->print();
  if (rightChild)
    rightChild->print();
}

std::vector<seq::ir::Value *> BETNode::matchOptFuncArgs(seq::ir::Module *M) {
  parseVars();
  std::vector<seq::ir::Value *> values = {};
  for (auto *v : vars) values.push_back(M->Nr<VarValue>(v));
  values.pop_back();  // Hack. Fix it.
  return values;
}

// /*
//  * Substitution optimizations
//  */

bool isCodimFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.internal.attributes.optimize_energy");
}

bool isUnary(CallInstr *callInstr) {
  return callInstr->numArgs() == 1;
}

int getOperator(CallInstr *callInstr) {
  auto *f = util::getFunc(callInstr->getCallee());
  auto instrName = f->getName();
  if (instrName.find("__add__") != std::string::npos)
    return BET_ADD_OP;
  if (instrName.find("__mul__") != std::string::npos)
    return BET_MUL_OP;
  if (instrName.find("__pow__") != std::string::npos)
    return BET_POW_OP;
  if (instrName.find("secure_reveal") != std::string::npos)
    return BET_REVEAL_OP;
  if (instrName.find("matmul") != std::string::npos)
    return BET_MATMUL_OP;
  if (instrName.find("norm") != std::string::npos)
    return BET_NORM_OP;
  return BET_OTHER_OP;
}

BETNode *parseArithmetic(CallInstr *callInstr) {
  auto *betNode = new BETNode();

  auto op = getOperator(callInstr);
  betNode->setOperator(op);

  if (isUnary(callInstr)) {
    auto *arg = callInstr->front();

    auto *argInstr = cast<CallInstr>(arg);
    auto *argConst = cast<IntConst>(arg);

    if (argConst)
      betNode->setLeftChild(new BETNode(arg->getId(), 0, true, argConst->getVal(), true));
    else if (!argInstr) {
      auto *childNode = new BETNode(arg->getUsedVariables().front()->getId());
      childNode->setVar(arg->getUsedVariables().front());
      betNode->setLeftChild(childNode);
    } else betNode->setLeftChild(parseArithmetic(argInstr));

    return betNode;
  }

  auto *lhs = callInstr->front();
  auto *rhs = callInstr->back();
  auto *lhsInstr = cast<CallInstr>(lhs);
  auto *rhsInstr = cast<CallInstr>(rhs);
  auto *lhsConst = cast<IntConst>(lhs);
  auto *rhsConst = cast<IntConst>(rhs);

  if (lhsConst)
    betNode->setLeftChild(new BETNode(lhs->getId(), 0, true, lhsConst->getVal(), true));
  else if (!lhsInstr) {
    auto *childNode = new BETNode(lhs->getUsedVariables().front()->getId());
    childNode->setVar(lhs->getUsedVariables().front());
    betNode->setLeftChild(childNode);
  } else betNode->setLeftChild(parseArithmetic(lhsInstr));

  if (rhsConst)
    betNode->setRightChild(
        new BETNode(rhs->getId(), 0, true, rhsConst->getVal(), true));
  else if (!rhsInstr) {
    auto *childNode = new BETNode(rhs->getUsedVariables().front()->getId());
    childNode->setVar(rhs->getUsedVariables().front());
    betNode->setRightChild(childNode);
  } else betNode->setRightChild(parseArithmetic(rhsInstr));

  return betNode;
}

BETNode *parseInstruction(seq::ir::Value *instruction) {
  auto *retIns = cast<ReturnInstr>(instruction);
  if (retIns)
    return NULL;

  auto *assIns = cast<AssignInstr>(instruction);
  if (!assIns)
    return NULL;

  auto *var = assIns->getLhs();
  auto *callInstr = cast<CallInstr>(assIns->getRhs());
  if (!callInstr)
    return NULL;

  if (var->getName().find("energy") != std::string::npos) {
    auto *betNode = parseArithmetic(callInstr);
    betNode->setVariableId(var->getId());
    return betNode;
  }

  return NULL;
}

CallInstr *optimizedCall(seq::ir::Module *M, BETNode *node) {
  std::string optFuncName = "optimize_norm";  // Hardcoded for now.
  std::vector<seq::ir::Value *> optFuncArgs = node->matchOptFuncArgs(M);
  std::vector<types::Type *> optFuncTypes = {};
  for (auto *v: optFuncArgs) optFuncTypes.push_back(v->getType());

  auto *numPyType = M->getOrRealizeType("NumPy", {}, "std.codim");
  auto *optFunc = M->getOrRealizeMethod(numPyType, optFuncName, optFuncTypes);
  assert(optFunc && "Opt func not found");

  return util::call(optFunc, optFuncArgs);
}

void convertInstructions(CallInstr *v, SeriesFlow *series, BETNode *node) {
  auto it = series->begin();
  while (it != series->end()) {
    auto *retIns = cast<ReturnInstr>(*it);
    if (retIns) {
      ++it;
      continue;
    }

    auto *assIns = cast<AssignInstr>(*it);
    if (!assIns) {
      ++it;
      continue;
    }

    bool isVar = assIns->getLhs()->getName().find("__var__") != std::string::npos;
    if (!isVar) {
      ++it;
      continue;
    }

    assIns->setRhs(optimizedCall(v->getModule(), node));
    break;  // Hack. Fix it.
  }
}

void ArithmeticsOptimizations::applyOptimizations(CallInstr *v) {
    auto *f = util::getFunc(v->getCallee());
    if (!isCodimFunc(f))
      return;

    auto *bf = cast<BodiedFunc>(f);
    auto *series = cast<SeriesFlow>(bf->getBody());
    BETNode *node = NULL;

    for (auto it = series->begin(); it != series->end(); ++it) {
      node = parseInstruction(*it);
      if (node) break;
    }

    if (!node) return;
    convertInstructions(v, series, node);
}

void ArithmeticsOptimizations::handle(CallInstr *v) { applyOptimizations(v); }

} // namespace seq
