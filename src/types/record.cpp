#include "seq/func.h"
#include "seq/base.h"
#include "seq/record.h"

using namespace seq;
using namespace llvm;

static std::string getNameFromTypes(std::vector<types::Type *> types)
{
	std::string name;

	for (auto *type : types)
		name += type->getName();

	name += "Recrod";
	return name;
}

types::RecordType::RecordType(std::vector<Type *> types) :
    Type(getNameFromTypes(types), BaseType::get(), SeqData::RECORD), types(std::move(types))
{
}

types::RecordType::RecordType(std::initializer_list<Type *> types) :
    Type(getNameFromTypes(types), BaseType::get(), SeqData::RECORD), types(types)
{
}

void types::RecordType::callCopy(BaseFunc *base,
                                 ValMap ins,
                                 ValMap outs,
                                 BasicBlock *block)
{
	IRBuilder<> builder(block);
	Value *rec = builder.CreateLoad(getSafe(ins, SeqData::RECORD));
	unpack(base, rec, outs, block);
}

void types::RecordType::callSerialize(BaseFunc *base,
                                      ValMap outs,
                                      Value *fp,
                                      BasicBlock *block)
{
	IRBuilder<> builder(block);
	Value *rec = builder.CreateLoad(getSafe(outs, SeqData::RECORD));

	for (int i = 0; i < types.size(); i++) {
		auto subOuts = std::make_shared<std::map<SeqData, Value *>>(*new std::map<SeqData, Value *>());
		Value *elem = builder.CreateExtractValue(rec, i);
		types[i]->unpack(base, elem, subOuts, block);
		types[i]->callSerialize(base, subOuts, fp, block);
	}
}

void types::RecordType::callDeserialize(BaseFunc *base,
                                        ValMap outs,
                                        Value *fp,
                                        BasicBlock *block)
{
	LLVMContext& context = block->getContext();
	IRBuilder<> builder(block);
	Value *rec = UndefValue::get(getLLVMType(context));

	for (int i = 0; i < types.size(); i++) {
		auto subOuts = std::make_shared<std::map<SeqData, Value *>>(*new std::map<SeqData, Value *>());
		types[i]->callDeserialize(base, subOuts, fp, block);
		Value *elem = types[i]->pack(base, subOuts, block);
		rec = builder.CreateInsertValue(rec, elem, i);
	}

	unpack(base, rec, outs, block);
}

static seq_int_t getIdxSafe(Value *idx, const seq_int_t max)
{
	if (auto *constIdx = dyn_cast<ConstantInt>(idx)) {
		const seq_int_t idxReal = constIdx->getSExtValue();

		if (idxReal < 1 || idxReal > max)
			throw exc::SeqException("index into record out of bounds");

		return idxReal - 1;  // 1-based to 0-based
	} else {
		throw exc::SeqException("index into record must be constant");
	}
}

void types::RecordType::codegenIndexLoad(BaseFunc *base,
                                         ValMap outs,
                                         BasicBlock *block,
                                         Value *ptr,
                                         Value *idx)
{
	const seq_int_t idxReal = getIdxSafe(idx, (seq_int_t)types.size());
	Type *type = types[idxReal];

	LLVMContext& context = base->getContext();
	BasicBlock *preambleBlock = base->getPreamble();
	IRBuilder<> builder(block);

	Value *recPtr = makeAlloca(ptr->getType(), preambleBlock);
	builder.CreateStore(ptr, recPtr);
	Value *elemPtr = builder.CreateGEP(recPtr,
	                                   {ConstantInt::get(IntegerType::getInt32Ty(context), 0),
	                                    ConstantInt::get(IntegerType::getInt32Ty(context), (uint64_t)idxReal)});
	type->codegenLoad(base, outs, block, elemPtr, zeroLLVM(context));
}

void types::RecordType::codegenIndexStore(BaseFunc *base,
                                          ValMap outs,
                                          BasicBlock *block,
                                          Value *ptr,
                                          Value *idx)
{
	const seq_int_t idxReal = getIdxSafe(idx, (seq_int_t)types.size());
	Type *type = types[idxReal];

	LLVMContext& context = base->getContext();
	BasicBlock *preambleBlock = base->getPreamble();
	IRBuilder<> builder(block);

	Value *recPtr = makeAlloca(ptr->getType(), preambleBlock);
	builder.CreateStore(ptr, recPtr);
	Value *elemPtr = builder.CreateGEP(recPtr,
	                                   {ConstantInt::get(IntegerType::getInt32Ty(context), 0),
	                                    ConstantInt::get(IntegerType::getInt32Ty(context), (uint64_t)idxReal)});
	type->codegenStore(base, outs, block, elemPtr, zeroLLVM(context));
}

bool types::RecordType::isGeneric(Type *type) const
{
	return dynamic_cast<types::RecordType *>(type) != nullptr;
}

types::Type *types::RecordType::getBaseType(seq_int_t idx) const
{
	if (idx < 1 || idx > (seq_int_t)types.size())
		throw exc::SeqException("invalid index into Record (must be constant and in-bounds)");

	return types[idx - 1];
}

Type *types::RecordType::getLLVMType(LLVMContext& context) const
{
	llvm::StructType *recStruct = StructType::create(context, "rec_t");
	std::vector<llvm::Type *> body;
	for (auto& type : types)
		body.push_back(type->getLLVMType(context));

	recStruct->setBody(body);
	return recStruct;
}

seq_int_t types::RecordType::size(Module *module) const
{
	std::unique_ptr<DataLayout> layout(new DataLayout(module));
	return layout->getTypeAllocSize(getLLVMType(module->getContext()));
}

types::RecordType& types::RecordType::of(std::initializer_list<std::reference_wrapper<Type>> types) const
{
	std::vector<Type *> typesPtr;
	for (auto& type : types)
		typesPtr.push_back(&type.get());

	return *RecordType::get(typesPtr);
}

types::RecordType *types::RecordType::get(std::vector<Type *> types)
{
	return new RecordType(std::move(types));
}

types::RecordType *types::RecordType::get(std::initializer_list<Type *> types)
{
	return new RecordType(types);
}
