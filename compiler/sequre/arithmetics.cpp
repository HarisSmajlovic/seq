#include "arithmetics.h"
#include "sir/util/cloning.h"
#include "sir/util/irtools.h"
#include "sir/util/matching.h"
#include <iterator>
#include <math.h>

namespace seq {

using namespace ir;

const std::string secureContainerTypeName = "SharedTensor";

/*
 * Binary expression tree
 */
const int BET_ADD_OP = 1;
const int BET_MUL_OP = 2;
const int BET_POW_OP = 3;
const int BET_REVEAL_OP = 4;
const int BET_OTHER_OP = 5;

class BETNode {
  int64_t value;
  Var *variable;
  types::Type *nodeType;
  int op;
  BETNode *leftChild;
  BETNode *rightChild;
  bool expanded;
  bool constant;

public:
  BETNode();
  BETNode(Var *variable);
  BETNode(Var *variable, int op, bool expanded, int64_t value, bool constant);
  BETNode(Var *variable, int op, BETNode *leftChild, BETNode *rightChild,
          bool expanded, int64_t value, bool constant);
  ~BETNode() {
    if (leftChild)
      delete leftChild;
    if (rightChild)
      delete rightChild;
  }

  void setVariable(Var *variable) { this->variable = variable; }
  void setOperator(int op) { this->op = op; }
  void setLeftChild(BETNode *leftChild) { this->leftChild = leftChild; }
  void setRightChild(BETNode *rightChild) { this->rightChild = rightChild; }
  void setExpanded() { expanded = true; }
  Var *getVariable() const { return variable; }
  int getVariableId() const { return variable ? variable->getId() : 0; }
  int getOperator() const { return op; }
  int64_t getValue() const { return value; }
  BETNode *getLeftChild() const { return leftChild; }
  BETNode *getRightChild() const { return rightChild; }
  bool isExpanded() const { return expanded; }
  bool isLeaf() const { return !leftChild && !rightChild; }
  bool isAdd() const { return op == BET_ADD_OP; }
  bool isMul() const { return op == BET_MUL_OP; }
  bool isPow() const { return op == BET_POW_OP; }
  bool isConstant() const { return constant; }
  bool isSameLeaf(BETNode *) const;
  void replace(BETNode *);
  BETNode *copy() const;
  void print(int) const;

  types::Type *getType();
  std::string const getOperatorIRName() const;
};

BETNode::BETNode()
    : value(1), variable(nullptr), nodeType(nullptr), op(0), leftChild(nullptr), rightChild(nullptr),
      expanded(false), constant(false) {}

BETNode::BETNode(Var *variable)
    : value(1), variable(variable), op(0), leftChild(nullptr), rightChild(nullptr),
      expanded(false), constant(false) { if (variable) nodeType = variable->getType(); else nodeType = nullptr; }

BETNode::BETNode(Var *variable, int op, bool expanded, int64_t value, bool constant)
    : value(value), variable(variable), op(op), leftChild(nullptr),
      rightChild(nullptr), expanded(expanded), constant(constant) { if (variable) nodeType = variable->getType(); else nodeType = nullptr; }

BETNode::BETNode(Var *variable, int op, BETNode *leftChild, BETNode *rightChild,
                 bool expanded, int64_t value, bool constant)
    : value(value), variable(variable), op(op), leftChild(leftChild),
      rightChild(rightChild), expanded(expanded), constant(constant) { if (variable) nodeType = variable->getType(); else nodeType = nullptr; }

bool BETNode::isSameLeaf(BETNode *other) const {
  if (isLeaf() && other->isLeaf()) {
    if (isConstant() && other->isConstant())
      return getValue() == other->getValue();
    
    assert(variable && "BET leaf is neither constant nor variable. (This is internal bug within IR optimizations. Please report it to code owners.)");

    int varId = getVariableId();
    int otherVarId = getVariableId();
    if (varId && otherVarId)
      return varId == otherVarId;
  }

  return false;
}

void BETNode::replace(BETNode *other) {
  variable = other->getVariable();
  op = other->getOperator();
  leftChild = other->getLeftChild();
  rightChild = other->getRightChild();
  expanded = other->isExpanded();
  value = other->getValue();
  constant = other->isConstant();
}

BETNode *BETNode::copy() const {
  auto *newNode = new BETNode(variable, op, expanded, value, constant);
  auto *lc = getLeftChild();
  auto *rc = getRightChild();
  if (lc)
    newNode->setLeftChild(lc->copy());
  if (rc)
    newNode->setRightChild(rc->copy());
  return newNode;
}

void BETNode::print(int level = 0) const {
  for (int i=0; i < level; ++i)
    std::cout << "    ";
  
  std::cout << op << " " << getVariableId()
            << (constant ? " Is constant " : " Not constant ") << value << std::endl;
  
  if (leftChild)
    leftChild->print(level + 1);
  if (rightChild)
    rightChild->print(level + 1);
}

types::Type *BETNode::getType() {
  if (!nodeType) {
    if (isConstant())
      nodeType = new types::IntType();
    else if (isLeaf())
      nodeType = variable->getType();
    else
      nodeType = getLeftChild()->getType();
  }
  
  assert(nodeType);
  return nodeType;
};

std::string const BETNode::getOperatorIRName() const {
  if (isAdd()) return "__add__";
  if (isMul()) return "__mul__";
  if (isPow()) return "__pow__";
  assert(false && "BET node operator not supported in IR optimizations.");
};

class BET {
  std::unordered_map<int, BETNode *> roots;
  std::vector<int> stopVarIds;
  std::set<int> vars;
  std::vector<BETNode *> polynomials;
  std::vector<std::vector<int64_t>> pascalMatrix;
  bool treeAltered;

public:
  BET() : treeAltered(false) {}
  ~BET() {
    auto *root = this->root();
    if (root)
      delete root;
  }

  int getVarsSize() const { return vars.size(); }
  void addRoot(BETNode *betNode) { roots[betNode->getVariableId()] = betNode; }
  void addRoot(Var *, int);
  void addNode(BETNode *);
  void addStopVar(int varId) { stopVarIds.insert(stopVarIds.begin(), varId); }
  void formPolynomials();
  void parseVars(BETNode *);
  BETNode *root();
  BETNode *polyRoot() const;
  BETNode *getNextPolyNode();
  std::vector<BETNode*> generateFactorizationTrees(int);
  std::vector<int64_t> extractCoefficents(BETNode *) const;
  std::vector<int64_t> extractExponents(BETNode *) const;
  std::set<int> extractVars(BETNode *) const;
  std::vector<std::vector<int64_t>> getPascalMatrix() const { return pascalMatrix; }

  bool expandLvl(BETNode *);
  bool reduceLvl(BETNode *);
  void expandAll(BETNode *);
  void reduceAll(BETNode *);

  void escapePows(BETNode *);

private:
  void addVar(int varId) { vars.insert(varId); }
  void expandNode(BETNode *);
  void expandPow(BETNode *);
  void expandMul(BETNode *);
  void collapseMul(BETNode *);
  void formPolynomial(BETNode *);
  void extractCoefficents(BETNode *, std::vector<int64_t> &) const;
  void extractExponents(BETNode *, std::vector<int64_t> &) const;
  void extractVars(BETNode *, std::set<int> &) const;
  void parseExponents(BETNode *, std::map<int, int64_t> &) const;
  void updatePascalMatrix(int64_t);
  BETNode *getMulTree(BETNode *, BETNode *, int64_t, int64_t);
  BETNode *getPowTree(BETNode *, BETNode *, int64_t, int64_t);
  int64_t parseCoefficient(BETNode *) const;
  int64_t getBinomialCoefficient(int64_t, int64_t);
  std::vector<int64_t> getPascalRow(int64_t);
};

void BET::expandNode(BETNode *betNode) {
  if (betNode->isExpanded())
    return;

  int op = betNode->getOperator();
  if (!op) {
    auto search = roots.find(betNode->getVariableId());
    if (search != roots.end())
      betNode->replace(search->second);
  } else {
    expandNode(betNode->getLeftChild());
    expandNode(betNode->getRightChild());
  }

  betNode->setExpanded();
}

void BET::expandPow(BETNode *betNode) {
  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  assert(rc->isConstant() &&
         "Sequre polynomial optimization expects each exponent to be a constant.");

  if (lc->isMul()) {
    treeAltered = true;
    betNode->setOperator(BET_MUL_OP);
    lc->setOperator(BET_POW_OP);
    auto *newPowNode =
        new BETNode(nullptr, BET_POW_OP, lc->getRightChild(), rc, true, 1, false);
    betNode->setRightChild(newPowNode);
    lc->setRightChild(rc->copy());
    return;
  }

  if (lc->isAdd()) {
    treeAltered = true;
    auto *v1 = lc->getLeftChild();
    auto *v2 = lc->getRightChild();

    auto *powTree = getPowTree(v1, v2, rc->getValue(), 0);

    betNode->setOperator(BET_ADD_OP);
    delete lc;
    betNode->setLeftChild(powTree->getLeftChild());
    delete rc;
    betNode->setRightChild(powTree->getRightChild());
    return;
  }
}

void BET::expandMul(BETNode *betNode) {
  treeAltered = true;

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  auto *addNode = lc->isAdd() ? lc : rc;
  auto *otherNode = lc->isAdd() ? rc : lc;
  betNode->setOperator(BET_ADD_OP);
  addNode->setOperator(BET_MUL_OP);
  auto *newMulNode =
      new BETNode(nullptr, BET_MUL_OP, addNode->getRightChild(), otherNode, true, 1, false);
  if (lc == otherNode)
    betNode->setLeftChild(newMulNode);
  if (rc == otherNode)
    betNode->setRightChild(newMulNode);
  addNode->setRightChild(otherNode->copy());
}

void BET::collapseMul(BETNode *betNode) {
  treeAltered = true;

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  auto *llc = lc->getLeftChild();
  auto *rlc = lc->getRightChild();
  auto *lrc = rc->getLeftChild();
  auto *rrc = rc->getRightChild();
  
  BETNode *collapseNode;
  BETNode *otherNode1, *otherNode2;

  if (llc->isSameLeaf(lrc)) {
    collapseNode = llc;
    otherNode1 = rlc;
    otherNode2 = rrc;
  } else if (llc->isSameLeaf(rrc)) {
    collapseNode = llc;
    otherNode1 = rlc;
    otherNode2 = lrc;
  } else if (rlc->isSameLeaf(lrc)) {
    collapseNode = rlc;
    otherNode1 = llc;
    otherNode2 = rrc;
  } else if (rlc->isSameLeaf(rrc)) {
    collapseNode = rlc;
    otherNode1 = llc;
    otherNode2 = lrc;
  } else {
    throw 1; // TODO: Remove after isSameLeaf is replaced with isSameSubTree
  }
  
  betNode->setOperator(BET_MUL_OP);
  lc->setOperator(BET_ADD_OP);
  lc->setLeftChild(otherNode1);
  lc->setRightChild(otherNode2);
  
  rc->replace(collapseNode);
}

void BET::formPolynomial(BETNode *betNode) {
  if (betNode->isLeaf())
    return;

  if (betNode->isPow()) {
    expandPow(betNode);
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();
  if (!betNode->isMul() || !(lc->isAdd() || rc->isAdd())) {
    formPolynomial(lc);
    formPolynomial(rc);
    return;
  }

  expandMul(betNode);
}

void BET::extractCoefficents(BETNode *betNode, std::vector<int64_t> &coefficients) const {
  if (!(betNode->isAdd())) {
    coefficients.push_back(parseCoefficient(betNode));
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();
  extractCoefficents(lc, coefficients);
  extractCoefficents(rc, coefficients);
}

void BET::extractExponents(BETNode *betNode, std::vector<int64_t> &exponents) const {
  if (!(betNode->isAdd())) {
    std::map<int, int64_t> termExponents;
    for (auto varId : vars)
      termExponents[varId] = 0;
    parseExponents(betNode, termExponents);
    for (auto e : termExponents)
      exponents.push_back(e.second);
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();
  extractExponents(lc, exponents);
  extractExponents(rc, exponents);
}

void BET::extractVars(BETNode *betNode, std::set<int> &vars) const {
  if (betNode->isConstant())
    return;
  if (betNode->isLeaf()) {
    vars.insert(betNode->getVariableId());
    return;
  }
    
  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();
  extractVars(lc, vars);
  extractVars(rc, vars);
}

void BET::parseExponents(BETNode *betNode, std::map<int, int64_t> &termExponents) const {
  if (betNode->isConstant())
    return;
  if (betNode->isLeaf()) {
    termExponents[betNode->getVariableId()]++;
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  if (betNode->isPow() && !lc->isConstant()) {
    assert(rc->isConstant() &&
           "Sequre polynomial optimization expects each exponent to be a constant.");
    termExponents[lc->getVariableId()] += rc->getValue();
    return;
  }

  parseExponents(lc, termExponents);
  parseExponents(rc, termExponents);
}

void BET::updatePascalMatrix(int64_t n) {
  for (auto i = pascalMatrix.size(); i < n + 1; ++i) {
    auto newRow = std::vector<int64_t>(i + 1);
    for (auto j = 0; j < i + 1; ++j)
      newRow[j] = (j == 0 || j == i)
                      ? 1
                      : (pascalMatrix[i - 1][j - 1] + pascalMatrix[i - 1][j]);
    pascalMatrix.push_back(newRow);
  }
}

BETNode *BET::getMulTree(BETNode *v1, BETNode *v2, int64_t constant, int64_t iter) {
  auto *pascalNode =
      new BETNode(nullptr, 0, true, getBinomialCoefficient(constant, iter), true);
  auto *leftConstNode = new BETNode(nullptr, 0, true, constant - iter, true);
  auto *rightConstNode = new BETNode(nullptr, 0, true, iter, true);
  auto *leftPowNode =
      new BETNode(nullptr, BET_POW_OP, v1->copy(), leftConstNode, true, 1, false);
  auto *rightPowNode =
      new BETNode(nullptr, BET_POW_OP, v2->copy(), rightConstNode, true, 1, false);
  auto *rightMulNode =
      new BETNode(nullptr, BET_MUL_OP, leftPowNode, rightPowNode, true, 1, false);

  return new BETNode(nullptr, BET_MUL_OP, pascalNode, rightMulNode, true, 1, false);
}

BETNode *BET::getPowTree(BETNode *v1, BETNode *v2, int64_t constant, int64_t iter) {
  auto *newMulNode = getMulTree(v1, v2, constant, iter);

  if (constant == iter)
    return newMulNode;

  auto *newAddNode = new BETNode(nullptr, BET_ADD_OP, true, 1, false);

  newAddNode->setLeftChild(newMulNode);
  newAddNode->setRightChild(getPowTree(v1, v2, constant, iter + 1));

  return newAddNode;
}

int64_t BET::parseCoefficient(BETNode *betNode) const {
  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  if (betNode->isPow()) {
    assert(lc->isLeaf() && "Pow expression should be at bottom of the polynomial tree");
    return (lc->isConstant() ? std::pow(lc->getValue(), rc->getValue()) : 1);
  }
  if (betNode->isConstant() || betNode->isLeaf()) {
    return betNode->getValue();
  }

  return parseCoefficient(lc) * parseCoefficient(rc);
}

int64_t BET::getBinomialCoefficient(int64_t n, int64_t k) {
  auto pascalRow = getPascalRow(n);
  return pascalRow[k];
}

std::vector<int64_t> BET::getPascalRow(int64_t n) {
  if (n >= pascalMatrix.size())
    updatePascalMatrix(n);

  return pascalMatrix[n];
}

void BET::addRoot(Var* newVar, int oldVarId) {
  auto *oldNode = roots[oldVarId]->copy();
  oldNode->setVariable(newVar);
  roots[oldNode->getVariableId()] = oldNode;
}

void BET::addNode(BETNode *betNode) {
  expandNode(betNode);
  addRoot(betNode);
}

void BET::formPolynomials() {
  for (int stopVarId : stopVarIds) {
    auto *polyRoot = roots[stopVarId]->copy();
    do {
      treeAltered = false;
      formPolynomial(polyRoot);
    } while (treeAltered);
    polynomials.push_back(polyRoot);
  }
}

void BET::parseVars(BETNode *betNode) {
  if (betNode->isConstant())
    return;
  if (betNode->isLeaf()) {
    addVar(betNode->getVariableId());
    return;
  }

  parseVars(betNode->getLeftChild());
  parseVars(betNode->getRightChild());
}

BETNode *BET::root() {
  if (!stopVarIds.size())
    return nullptr;

  auto stopVarId = stopVarIds.back();
  auto search = roots.find(stopVarId);
  if (search == roots.end())
    return nullptr;

  return roots[stopVarId];
}

BETNode *BET::polyRoot() const {
  if (!polynomials.size())
    return nullptr;

  return polynomials.back();
}

BETNode *BET::getNextPolyNode() {
  auto *polyNode = polyRoot();

  if (polyNode) {
    polynomials.pop_back();
    return polyNode;
  }

  return nullptr;
}

std::vector<int64_t> BET::extractCoefficents(BETNode *betNode) const {
  std::vector<int64_t> coefficients;
  extractCoefficents(betNode, coefficients);
  return coefficients;
}

std::vector<int64_t> BET::extractExponents(BETNode *betNode) const {
  std::vector<int64_t> exponents;
  extractExponents(betNode, exponents);
  return exponents;
}

std::set<int> BET::extractVars(BETNode *betNode) const {
  std::set<int> varIds;
  extractVars(betNode, varIds);
  return varIds;
}

void BET::escapePows(BETNode *node) {
  if (node->isLeaf())
    return;
  
  if (!node->isPow()) {
    escapePows(node->getLeftChild());
    escapePows(node->getRightChild());
    return;
  }

  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();

  assert(rc->isConstant() &&
         "Sequre factorization optimization expects each exponent to be a constant.");
  assert(rc->getValue() > 0 &&
         "Sequre factorization optimization expects each exponent to be positive.");

  auto *newMulNode = new BETNode(nullptr, BET_MUL_OP, lc, lc, false, 1, false);

  if (rc->getValue() == 1)
    newMulNode->setRightChild(new BETNode(nullptr, 0, true, 1, true));

  for (int i=0; i < rc->getValue() - 2; ++i)
    newMulNode = new BETNode(nullptr, BET_MUL_OP, lc, newMulNode->copy(), false, 1, false);

  node->replace(newMulNode);
}

std::vector<BETNode*> BET::generateFactorizationTrees(int upperLimit = 10) {
  BETNode *root = this->root()->copy();
  escapePows(root);
  reduceAll(root);

  std::vector<BETNode*> factorizations;
  for (int i=0; i != upperLimit; ++i) {
    factorizations.push_back(root->copy());
    if (!expandLvl(root)) break;
  }

  return factorizations;
}

bool BET::expandLvl(BETNode *node) {
  // TODO: Add support for operators other than + and *
  
  if (node->isLeaf())
    return false;

  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();
  if (!node->isMul() || !(lc->isAdd() || rc->isAdd())) {
    if (expandLvl(lc)) return true;
    return expandLvl(rc);
  }

  expandMul(node);
  return true;
}

bool BET::reduceLvl(BETNode *node) {
  // TODO: Add support for operators other than + and *

  if (node->isLeaf())
    return false;
  
  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();
  bool not_reducible = !node->isAdd() || !(lc->isMul() && rc->isMul());

  bool reducible = false;

  if (!lc->isLeaf() && !rc->isLeaf()) {
    auto *llc = lc->getLeftChild();
    auto *rlc = lc->getRightChild();
    auto *lrc = rc->getLeftChild();
    auto *rrc = rc->getRightChild();
  
    reducible = llc->isSameLeaf(lrc) || llc->isSameLeaf(rrc) || rlc->isSameLeaf(lrc) || rlc->isSameLeaf(rrc);
  }
  
  if (not_reducible or !reducible) {
    if (reduceLvl(lc)) return true;
    return reduceLvl(rc);
  }

  collapseMul(node);
  return true;
}

void BET::expandAll(BETNode *root) {
  while(expandLvl(root));
}

void BET::reduceAll(BETNode *root) {
  while(reduceLvl(root));
}

bool isArithmetic(int op) { return op && op < 4; }
bool isReveal(int op) { return op == 4; }

/*
 * Substitution optimizations
 */

bool isSequreFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.sequre");
}

bool isPolyOptFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.sequre_poly");
}

bool isBeaverOptFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.sequre_beaver");
}

bool isFactOptFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.opt_mat_arth");
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
  return BET_OTHER_OP;
}

types::Type *getTupleType(int n, types::Type *elemType, Module *M) {
  std::vector<types::Type *> tupleTypes;
  for (int i = 0; i != n; ++i)
    tupleTypes.push_back(elemType);
  return M->getTupleType(tupleTypes);
}

BETNode *parseArithmetic(CallInstr *callInstr) {
  // Arithmetics are binary
  auto *betNode = new BETNode();

  auto op = getOperator(callInstr);
  betNode->setOperator(op);

  auto *lhs = callInstr->front();
  auto *rhs = callInstr->back();
  auto *lhsInstr = cast<CallInstr>(lhs);
  auto *rhsInstr = cast<CallInstr>(rhs);
  auto *lhsConst = cast<IntConst>(lhs);
  auto *rhsConst = cast<IntConst>(rhs);

  if (lhsConst)
    betNode->setLeftChild(new BETNode(cast<Var>(lhs), 0, true, lhsConst->getVal(), true));
  else if (!lhsInstr) {
    betNode->setLeftChild(new BETNode(lhs->getUsedVariables().front()));
  } else
    betNode->setLeftChild(parseArithmetic(lhsInstr));
  
  if (rhsConst)
    betNode->setRightChild(
        new BETNode(cast<Var>(rhs), 0, true, rhsConst->getVal(), true));
  else if (!rhsInstr) {
    betNode->setRightChild(new BETNode(rhs->getUsedVariables().front()));
  } else
    betNode->setRightChild(parseArithmetic(rhsInstr));

  return betNode;
}

void parseInstruction(seq::ir::Value *instruction, BET *bet) {
  auto *retIns = cast<ReturnInstr>(instruction);
  if (retIns) {
    auto vars = retIns->getValue()->getUsedVariables();
    bet->addStopVar(vars.front()->getId());
    return;
  }

  auto *assIns = cast<AssignInstr>(instruction);
  if (!assIns)
    return;

  auto *var = assIns->getLhs();
  auto *callInstr = cast<CallInstr>(assIns->getRhs());
  if (!callInstr)
    return;
  
  auto op = getOperator(callInstr);
  if (isArithmetic(op)) {
    auto *betNode = parseArithmetic(callInstr);
    betNode->setVariable(var);
    bet->addNode(betNode);
  } else if (isReveal(op)) {
    bet->addRoot(var, util::getVar(callInstr->back())->getId());
    bet->addStopVar(var->getId());
  }
}

CallInstr *nextPolynomialCall(CallInstr *v, BodiedFunc *bf, BET *bet) {
  auto polyNode = bet->getNextPolyNode();
  auto coefs = bet->extractCoefficents(polyNode);
  auto exps = bet->extractExponents(polyNode);
  auto vars = bet->extractVars(polyNode);

  auto *M = v->getModule();
  auto *self = M->Nr<VarValue>(bf->arg_front());
  auto *selfType = self->getType();
  auto *funcType = cast<types::FuncType>(bf->getType());
  auto *returnType = funcType->getReturnType();
  auto *inputsType = getTupleType(vars.size(), returnType, M);
  auto *coefsType = getTupleType(coefs.size(), M->getIntType(), M);
  auto *expsType = getTupleType(exps.size(), M->getIntType(), M);

  auto *evalPolyFunc = M->getOrRealizeMethod(
      selfType, "secure_evalp", {selfType, inputsType, coefsType, expsType});
  assert(evalPolyFunc && "secure_evalp not found in provided MPC class");

  std::vector<Value *> inputArgs;
  for (auto it = bf->arg_begin(); it != bf->arg_end(); ++it) {
    if (vars.find((*it)->getId()) == vars.end())
      continue;
    auto *arg = M->Nr<VarValue>(*it);
    inputArgs.push_back(arg);
  }
  std::vector<Value *> coefsArgs;
  for (auto e : coefs)
    coefsArgs.push_back(M->getInt(e));
  std::vector<Value *> expsArgs;
  for (auto e : exps)
    expsArgs.push_back(M->getInt(e));

  auto *inputArg = util::makeTuple(inputArgs, M);
  auto *coefsArg = util::makeTuple(coefsArgs, M);
  auto *expsArg = util::makeTuple(expsArgs, M);

  return util::call(evalPolyFunc, {self, inputArg, coefsArg, expsArg});
}

Value *generateExpression(Module *M, BETNode *node) {
  if (node->isLeaf()) {
    auto *var = node->getVariable();
    assert(var);

    auto *arg = M->Nr<VarValue>(var);
    assert(arg);

    return arg;
  }
  
  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();
  assert(lc);
  assert(rc);

  auto *lopType = lc->getType();
  auto *ropType = rc->getType();
  auto *opFunc = M->getOrRealizeMethod(lopType, node->getOperatorIRName(), {lopType, ropType});

  std::string const errMsg = node->getOperatorIRName() + " not found in type " + lopType->getName();
  assert(opFunc && errMsg.c_str());

  auto *lop = generateExpression(M, lc);
  assert(lop);
  auto *rop = generateExpression(M, rc);
  assert(rop);

  auto *callIns = util::call(opFunc, {lop, rop});
  assert(callIns);
  auto *actualCallIns = callIns->getActual();
  assert(actualCallIns);

  return actualCallIns;
}

void convertInstructions(CallInstr *v, BodiedFunc *bf, SeriesFlow *series, BET *bet) {
  auto it = series->begin();
  while (it != series->end()) {
    auto *retIns = cast<ReturnInstr>(*it);
    if (retIns) {
      retIns->setValue(nextPolynomialCall(v, bf, bet));
      ++it;
      continue;
    }

    auto *assIns = cast<AssignInstr>(*it);
    if (!assIns) {
      ++it;
      continue;
    }

    auto *callInstr = cast<CallInstr>(assIns->getRhs());
    if (!callInstr) {
      ++it;
      continue;
    }

    auto op = getOperator(callInstr);
    if (isArithmetic(op)) {
      it = series->erase(it);
      continue;
    }

    if (isReveal(op)) {
      callInstr->setArgs({callInstr->front(), nextPolynomialCall(v, bf, bet)});
      ++it;
      continue;
    }
  }
}

void routeFactorizations(CallInstr *v, BodiedFunc *bf, SeriesFlow *series, std::vector<BETNode*> factorizationTrees) {
  auto it = series->begin();
  while (it != series->end()) {
    auto *retIns = cast<ReturnInstr>(*it);
    if (!retIns) {
      ++it;
      continue;
    }

    auto *debugCallVal = generateExpression(v->getModule(), factorizationTrees[0]);
    assert(debugCallVal);
    auto *debugCall = cast<CallInstr>(debugCallVal);
    assert(debugCall);
    retIns->setValue(debugCall);
    ++it;
  }
}

BET* parseBET(SeriesFlow *series) {
  auto *bet = new BET();
  for (auto it = series->begin(); it != series->end(); ++it)
    parseInstruction(*it, bet);
  
  bet->parseVars(bet->root());

  return bet;
}

void ArithmeticsOptimizations::applyFactorizationOptimizations(CallInstr *v) {
  auto *f = util::getFunc(v->getCallee());
  if (!isFactOptFunc(f))
    return;

  auto *bf = cast<BodiedFunc>(f);
  auto *series = cast<SeriesFlow>(bf->getBody());

  auto *bet = parseBET(series);
  auto factorizationTrees = bet->generateFactorizationTrees();

  routeFactorizations(v, bf, series, factorizationTrees);
}

void ArithmeticsOptimizations::applyPolynomialOptimizations(CallInstr *v) {
  auto *f = util::getFunc(v->getCallee());
  if (!isPolyOptFunc(f))
    return;

  auto *bf = cast<BodiedFunc>(f);
  auto *series = cast<SeriesFlow>(bf->getBody());

  auto *bet = parseBET(series);
  bet->formPolynomials();

  convertInstructions(v, bf, series, bet);
}

void ArithmeticsOptimizations::applyBeaverOptimizations(CallInstr *v) {
  auto *pf = getParentFunc();
  if (!isSequreFunc(pf) && !isBeaverOptFunc(pf))
    return;
  auto *f = util::getFunc(v->getCallee());
  if (!f)
    return;
  bool isEq = f->getName().find("__eq__") != std::string::npos;
  bool isGt = f->getName().find("__gt__") != std::string::npos;
  bool isLt = f->getName().find("__lt__") != std::string::npos;
  bool isAdd = f->getName().find("__add__") != std::string::npos;
  bool isSub = f->getName().find("__sub__") != std::string::npos;
  bool isMul = f->getName().find("__mul__") != std::string::npos;
  bool isDiv = f->getName().find("__truediv__") != std::string::npos;
  bool isPow = f->getName().find("__pow__") != std::string::npos;
  if (!isEq && !isGt && !isLt && !isAdd && !isSub && !isMul && !isPow && !isDiv)
    return;

  auto *M = v->getModule();
  auto *self = M->Nr<VarValue>(pf->arg_front());
  auto *selfType = self->getType();
  auto *lhs = v->front();
  auto *rhs = v->back();
  auto *lhsType = lhs->getType();
  auto *rhsType = rhs->getType();

  bool isSqrtInv = false;
  if (isDiv) {  // Special case where 1 / sqrt(x) is called
    auto *sqrtInstr = cast<CallInstr>(rhs);
    if (sqrtInstr) {
      auto *sqrtFunc = util::getFunc(sqrtInstr->getCallee());
      if (sqrtFunc)
        isSqrtInv = sqrtFunc->getName().find("sqrt") != std::string::npos;
    }
  }

  bool lhs_is_secure_container = lhsType->getName().find(secureContainerTypeName) != std::string::npos;
  bool rhs_is_secure_container = rhsType->getName().find(secureContainerTypeName) != std::string::npos;

  if (!lhs_is_secure_container and !rhs_is_secure_container)
    return;

  bool lhs_is_int = lhsType->is(M->getIntType());
  bool rhs_is_int = rhsType->is(M->getIntType());

  if (isMul && lhs_is_int)
    return;
  if (isMul && rhs_is_int)
    return;
  if (isDiv && lhs_is_int && !isSqrtInv)
    return;
  if (isPow && lhs_is_int)
    return;
  if (isPow && !rhs_is_int)
    return;

  std::string methodName =
    isEq      ? "secure_eq"       :
    isGt      ? "secure_gt"       :
    isLt      ? "secure_lt"       :
    isAdd     ? "secure_add"      :
    isSub     ? "secure_sub"      :
    isMul     ? "secure_mult"     :
    isSqrtInv ? "secure_sqrt_inv" :
    isDiv     ? "secure_div"      :
    isPow     ? "secure_pow"      :
    "invalid_operation";
  if (!isBeaverOptFunc(pf) && (isMul || isPow || isSqrtInv))
    methodName += "_no_cache";

  if (isSqrtInv) {
    rhs = cast<CallInstr>(rhs)->back();
    rhsType = rhs->getType();
  }

  auto *sequreInternalType = M->getOrRealizeType("Internal", {}, "std.sequre.stdlib.internal");
  auto *method =
      M->getOrRealizeMethod(sequreInternalType, methodName, {selfType, lhsType, rhsType});
  
  if (!method)
    return;

  auto *func = util::call(method, {self, lhs, rhs});
  v->replaceAll(func);
}

void ArithmeticsOptimizations::applyOptimizations(CallInstr *v) {
  applyPolynomialOptimizations(v);
  applyBeaverOptimizations(v);
  applyFactorizationOptimizations(v);
}

void ArithmeticsOptimizations::handle(CallInstr *v) { applyOptimizations(v); }

} // namespace seq
