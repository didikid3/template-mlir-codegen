#include "Execution/JIT.hpp"
#include "Templates/Custom/Utils.hpp"
#include "Templates/PredefinedPatterns.hpp"

#include "llvm/ADT/Hashing.h"
#include "mlir/IR/Iterators.h"

#include <algorithm>
#include <dlfcn.h>
#include <unordered_map>
namespace {
void assignValuesInTopologicalOrder(JIT& jit, uint16_t cgDepth, mlir::Operation::operand_range operands, mlir::Block::BlockArgListType args)
{
    std::vector<std::pair<JIT::Storage, JIT::Storage>> assignments;
    auto it = operands.begin();
    for (const auto& arg : args) {
        const auto from = jit.getStorage(*it);
        const auto to = jit.getOrCreateSlot(arg);
        ASSURE_FALSE(from.same(to));
        assignments.emplace_back(from, to);
        it++;
    }
    auto stackOffset = jit.stackOffset();
    while (!assignments.empty()) {
        const auto sizeBefore = assignments.size();
        for (auto it = assignments.cbegin(); it != assignments.cend(); it++) {
            // if the assignment target is not used as assignment source again, we can assign it
            if (std::find_if(assignments.cbegin(), assignments.cend(), [&](const auto& a) { return it->second.same(a.first); }) == assignments.cend()) {
                ASSURE_TRUE(it->second.isStack());
                jit.moveTo(cgDepth, it->first, it->second);
                assignments.erase(it);
                break;
            }
        }
        if (assignments.size() == sizeBefore) {
            // in this case we have a dependency cycle so we (try to) break it, by temporarily duplicating one value
            // and by that removing its dependency
            auto& assignment = assignments.front();
            const auto duplicateEntry = JIT::Storage::Stack(stackOffset, assignment.first.size());
            jit.moveTo(cgDepth, assignment.first, duplicateEntry);
            stackOffset += assignment.first.size().slots();
            assignment.first = duplicateEntry;
        }
    }
    jit.m_maxStackSpace = std::max(jit.m_maxStackSpace, stackOffset);
}

void generateCond(JIT& jit, uint16_t cgDepth, mlir::Operation::operand_range trueOps, mlir::Block* trueBlock, mlir::Operation::operand_range falseOps, mlir::Block* falseBlock)
{
#if defined(__x86_64__)
    // je 00000
    constexpr uint8_t CODE[] = {0x0f, 0x84};
    LOG_JIT(
        cgDepth,
        6,
        "CondBr");
    jit.insert(cgDepth, 2u, CODE);
    char* elseAddress = jit.memory(cgDepth, 4);
    assignValuesInTopologicalOrder(jit, cgDepth, trueOps, trueBlock->getArguments());
    JIT::Materializer<Br>().instantiate(jit, cgDepth);
    jit.insertBlockAddress(jit.memory(cgDepth) - 4, trueBlock);
    const uint64_t sizeFalse = reinterpret_cast<uint64_t>(jit.memory(cgDepth)) - reinterpret_cast<uint64_t>(elseAddress + 4);
    ASSURE(sizeFalse, <=, INT32_MAX);
    UnalignedWrite(elseAddress, static_cast<uint32_t>(sizeFalse));
    if (falseOps.empty()) {
        // for cases where we do not have to assign anything in the else branch, we can directly jump to the target
        jit.insertBlockAddress(elseAddress, falseBlock);
    } else {
        assignValuesInTopologicalOrder(jit, cgDepth, falseOps, falseBlock->getArguments());
        JIT::Materializer<Br>().instantiate(jit, cgDepth);
        jit.insertBlockAddress(jit.memory(cgDepth) - 4, falseBlock);
    }
#elif defined(__aarch64__)
    // tbz x20, #0, .else
    uint32_t* elseAddress = reinterpret_cast<uint32_t*>(jit.memory(cgDepth, 4));
    assignValuesInTopologicalOrder(jit, cgDepth, trueOps, trueBlock->getArguments());
    JIT::Materializer<Br>().instantiate(jit, cgDepth);
    jit.insertBlockAddress(jit.memory(cgDepth) - 4, trueBlock);
    const uint64_t sizeFalse = reinterpret_cast<uint64_t>(jit.memory(cgDepth)) - reinterpret_cast<uint64_t>(elseAddress);
    ASSURE(sizeFalse, <, (1 << 14));
    *elseAddress = TBZ_FIRST_ARGUMENT | ((sizeFalse >> 2) << 5);
    assignValuesInTopologicalOrder(jit, cgDepth, falseOps, falseBlock->getArguments());
    JIT::Materializer<Br>().instantiate(jit, cgDepth);
    jit.insertBlockAddress(jit.memory(cgDepth) - 4, falseBlock);
#endif
    jit.applyClobberMask(CLOBBER_SCRATCH_1 | ClobberMask(0b0000000001));
}
}

void CustomPattern<mlir::LLVM::SwitchOp>::Codegen(mlir::LLVM::SwitchOp& op, JIT& jit, [[maybe_unused]] uint16_t sourceDepth, uint16_t cgDepth)
{
    ASSURE(sourceDepth, <=, 2);
    if (!op.getCaseValues().has_value()) {
        // in case of no cases we just do a branch to the default target
        assignValuesInTopologicalOrder(jit, cgDepth, op.getDefaultOperands(), op.getDefaultDestination()->getArguments());
        JIT::Materializer<Br>().instantiate(jit, cgDepth);
        jit.insertBlockAddress(jit.memory(cgDepth) - 4, op.getDefaultDestination());
        return;
    }
    const bool hasOperands = std::any_of(op.getCaseOperands().begin(), op.getCaseOperands().end(), [](auto ops) { return !ops.empty(); }) || !op.getDefaultOperands().empty();
    std::vector<std::pair<uint64_t, std::pair<mlir::Block*, mlir::Operation::operand_range>>> cases;
    uint64_t min = static_cast<uint64_t>(-1);
    uint64_t max = 0;
    auto it1 = op.getCaseDestinations().begin();
    auto it2 = op.getCaseOperands().begin();
    for (const auto i : op.getCaseValues()->getValues<llvm::APInt>()) {
        const auto val = i.getZExtValue();
        min = std::min(val, min);
        max = std::max(val, max);
        cases.emplace_back(i.getZExtValue(), std::pair<mlir::Block*, mlir::Operation::operand_range>{*it1, *it2});
        it1++;
        it2++;
    }
    std::sort(cases.begin(), cases.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });
    const uint64_t range = max - min;

    std::vector<uint64_t> vec;
    jit.forceOnStack(cgDepth, jit.stackOffset(), {op.getValue()}, vec, vec.cend(), false);
    ASSURE_FALSE(vec.empty());
    const auto offset = static_cast<uint32_t>(vec.front());

    // 0. load value
#ifdef __aarch64__
    ASSURE(offset % 8, ==, 0);
    ASSURE(offset / 8, <=, (1ul << 12) - 1);
#endif
    ASM(MOV_STACK_TO_SCRATCH, offset);
    // 1. subtract the min (if != 0)
    const auto minValue = static_cast<uint32_t>(min);
    if (min != 0ul) {
        if (min <= MAX_IMM_SIZE) {
            ASM(SUB_SCRATCH, minValue);
        } else {
            ASM(MOV_CONST_TO_SCRATCH2, -min);
            ASM(ADD_SCRATCH_1_2);
        }
    }

    if (SUPPORT_SWITCH_TABLE && min != max && static_cast<double>(cases.size()) / range > 0.6 && cases.size() > 4 && !hasOperands) {
#ifdef __x86_64__
        // dense case with a fill factor above 80% and more than 4 entries -> build a direct mapping table
        // 2. if value > max -> default case (if it was actually smaller than the min, subtraction would have overflown)
        ASSURE(max - min, <=, UINT32_MAX);
        ASM(CMP_SCRATCH, static_cast<uint32_t>(max - min));
        {
            // ja default
            const uint8_t CODE[] = {0x0f, 0x87};
            jit.insert(cgDepth, 2, CODE);
        }
        jit.insertBlockAddress(jit.memory(cgDepth, 4), op.getDefaultDestination());
        // 3. build map
        const auto tableAddress = jit.memory(cgDepth + 1, 4 * (range + 1));
        INFO("jump table at " << reinterpret_cast<void*>(tableAddress));
        auto it = cases.cbegin();
        for (unsigned i = 0; i < (range + 1); i++) {
            const int32_t off = 4 * i;
            UnalignedWrite<int32_t>(tableAddress + 4 * i, off);
            if ((it->first - minValue) != i)
                jit.m_blocks.insertFixupLocation(op.getDefaultDestination(), tableAddress + 4 * i);
            else {
                jit.m_blocks.insertFixupLocation(it->second.first, tableAddress + 4 * i);
                it++;
            }
        }
        ASSURE_TRUE(it == cases.cend());

        // 4. jump to the respective entry
        const auto location = jit.memory(cgDepth) + 3;
        const int64_t relAddr = reinterpret_cast<int64_t>(tableAddress) - (reinterpret_cast<int64_t>(location) + 4);
        ASSURE(relAddr, >=, INT32_MIN);
        ASSURE(relAddr, <=, INT32_MAX);
        ASM(LEA_SCRATCH2, static_cast<int32_t>(relAddr));
        ASM(MOV_OFFSETED_TO_SCRATCH);
        ASM(ADD_SCRATCH_1_2);
        ASM(JMP_SCRATCH);
#endif
    } else {
        // apply binary search
        std::function<void(size_t, size_t)> searchInRange;
        searchInRange = [&](const auto minIndx, const auto maxIndx) {
            const auto mid = (minIndx + maxIndx) / 2;
            const uint64_t pivotElement = cases[mid].first - minValue;
            // less-equal to left, others to the right
            const bool eq = minIndx == maxIndx;
#if defined(__x86_64__)
            if (pivotElement <= MAX_IMM_SIZE) {
                ASM(CMP_SCRATCH, static_cast<uint32_t>(pivotElement));
            } else {
                ASM(MOV_CONST_TO_SCRATCH2, pivotElement);
                ASM(CMP_SCRATCH_1_2);
            }
            if (eq) {
                generateCond(
                    jit,
                    cgDepth,
                    op.getDefaultOperands(),
                    op.getDefaultDestination(),
                    cases[minIndx].second.second,
                    cases[minIndx].second.first);
                return;
            }
            {
                // ja *else
                const uint8_t CODE[] = {0x0f, 0x87};
                jit.insert(cgDepth, 2, CODE);
            }

            char* location = jit.memory(cgDepth, 4);
            searchInRange(minIndx, mid);
            const int64_t relAddr = reinterpret_cast<int64_t>(jit.memory(cgDepth)) - (reinterpret_cast<int64_t>(location) + 4);
            ASSURE(relAddr, >=, INT32_MIN);
            ASSURE(relAddr, <=, INT32_MAX);
            UnalignedWrite(location, static_cast<int32_t>(relAddr));
#elif defined(__aarch64__)
            if (pivotElement <= MAX_IMM_SIZE) {
                ASM(CMP_SCRATCH, static_cast<uint32_t>(pivotElement));
                if (eq)
                    ASM(SET_FIRST_ARG_EQ);
                else
                    ASM(SET_FIRST_ARG_LE);
            } else {
                ASM(MOV_CONST_TO_SCRATCH2, pivotElement);
                ASM(CMP_SCRATCH_1_2);
                if (eq)
                    ASM(SET_FIRST_ARG_EQ);
                else
                    ASM(SET_FIRST_ARG_LE);
            }
            if (eq) {
                generateCond(
                    jit,
                    cgDepth,
                    cases[minIndx].second.second,
                    cases[minIndx].second.first,
                    op.getDefaultOperands(),
                    op.getDefaultDestination());
                return;
            }
            char* elseAddress = jit.memory(cgDepth, 4);
            searchInRange(minIndx, mid);
            const int64_t sizeFalse = reinterpret_cast<int64_t>(jit.memory(cgDepth)) - reinterpret_cast<int64_t>(elseAddress);
            ASSURE(sizeFalse, <, (1 << 14));
            PatchInfo<TBZ_FIRST_ARGUMENT>::patch(reinterpret_cast<uint32_t*>(elseAddress), sizeFalse);
#endif
            searchInRange(mid + 1, maxIndx);
        };
        searchInRange(0, cases.size() - 1);

        // clobber rax, rcx
        jit.applyClobberMask(CLOBBER_SCRATCH);
    }
}

void CustomPattern<mlir::LLVM::BrOp>::Codegen(mlir::LLVM::BrOp& op, JIT& jit, [[maybe_unused]] uint16_t sourceDepth, uint16_t cgDepth)
{
    ASSURE(sourceDepth, <=, 2);
    assignValuesInTopologicalOrder(jit, cgDepth, op.getDestOperands(), op.getSuccessor()->getArguments());

    // insert a jump to the block address
    JIT::Materializer<Br>().instantiate(jit, cgDepth);
    jit.insertBlockAddress(jit.memory(cgDepth) - 4, op.getSuccessor());
}

void CustomPattern<mlir::LLVM::CondBrOp>::Codegen(mlir::LLVM::CondBrOp& op, JIT& jit, [[maybe_unused]] uint16_t sourceDepth, uint16_t cgDepth)
{
    ASSURE(sourceDepth, <=, 2);
    ASSURE(op.getCondition().getType().getIntOrFloatBitWidth(), ==, 1);
    const auto storage = jit.getStorage(op.getCondition());
    if (storage.isStack() && jit.isValid(storage.tmpRegister(), op.getCondition())) {
        jit.moveTo(cgDepth, JIT::Storage::Register(storage.tmpRegister()), JIT::Storage::Register(1));
    } else if (!storage.isRegister() || storage.getOffset() != 1) {
        jit.moveTo(cgDepth, storage, JIT::Storage::Register(1));
    }
#if defined(__x86_64__)
    LOG_JIT(
        cgDepth,
        sizeof(TEST_FIRST_ARGUMENT),
        "Cond",
        static_cast<uint32_t>(-1),
        llvm::cast<mlir::FileLineColLoc>(op->getLoc()).getLine());
    ASM(TEST_FIRST_ARGUMENT);
#endif
    generateCond(jit, cgDepth, op.getTrueDestOperands(), op.getTrueDest(), op.getFalseDestOperands(), op.getFalseDest());
}

void CustomPattern<mlir::LLVM::ConstantOp>::Codegen(mlir::LLVM::ConstantOp& op, JIT& jit, uint16_t, uint16_t)
{
    auto [val, size] = fillConstantFromAttr(jit, op.getValue(), op.getResult().getType());
    jit.setConstant(op.getResult(), size, std::move(val));
}

void CustomPattern<mlir::LLVM::UndefOp>::Codegen(mlir::LLVM::UndefOp& op, JIT& jit, uint16_t, uint16_t)
{
    const auto size = TypeSize::FromType(op.getResult().getType());
    const auto storage = jit.nextSlot<JIT::Storage::STACK>(size);
    jit.setStorage(op.getResult(), storage);
}

void CustomPattern<mlir::LLVM::GlobalOp>::Codegen(mlir::LLVM::GlobalOp& op, JIT& jit, uint16_t, uint16_t)
{
    if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
        // use local multi-map instead of global storage map of JIT as we can now have multiple entries
        std::unordered_multimap<mlir::Value, JIT::Storage, decltype([](const mlir::Value& val) { return mlir::hash_value(val); })> storedConstants;
        std::vector<std::pair<mlir::Value, JIT::Storage>> inserts;
        auto foreach_storage = [&](mlir::Operation* op, auto func) {
            auto result = op->getResult(0);
            // this would be a pretty useful assure to check that we do not miss anything
            // however, it might happen, that some values are not actually used (also possible in a transitive manner)
            // ASSURE(result.use_empty(), ||, storedConstants.count(result) > 0, << *op);
            inserts.clear();
            const auto [begin, end] = storedConstants.equal_range(result);
            std::for_each(begin, end, [&](const auto elem) {
                const auto storage = elem.second;
                ASSURE_TRUE(storage.isConstant());
                ASSURE(storage.size().bytes(), ==, TypeSize::FromType(result.getType()).bytes());
                func(storage);
            });
            for (auto insert : inserts) {
                storedConstants.emplace(insert);
            }
        };
        // go through all ops twice:
        // 1. assign location to store the value for each of the ops (in reverse order)
        // 2. do the actual assignments
        for (auto& o : mlir::ReverseIterator::makeIterable(op.getBodyRegion().getBlocks().front())) {
            llvm::TypeSwitch<mlir::Operation*, void>(&o)
                .Case([&](mlir::LLVM::NullOp nullOp) {
                    foreach_storage(nullOp, [](auto) {});
                })
                .Case([&](mlir::LLVM::ConstantOp constOp) {
                    foreach_storage(constOp, [](auto) {});
                })
                .Case([&](mlir::arith::ConstantOp constOp) {
                    foreach_storage(constOp, [](auto) {});
                })
                .Case([&](mlir::LLVM::UndefOp undefOp) {
                    foreach_storage(undefOp, [](auto) {});
                })
                .Case([&](mlir::LLVM::InsertValueOp insertOp) {
                    foreach_storage(insertOp, [&](const auto container) {
                        const auto indices = insertOp.getPosition().vec();
                        const auto off = accumulateOffsets(insertOp.getContainer().getType(), indices.cbegin(), indices.cend());
                        const auto valueSize = TypeSize::FromType(insertOp.getValue().getType());
                        ASSURE(off + valueSize.bytes(), <=, container.size().bytes());
                        inserts.emplace_back(insertOp.getContainer(), container);
                        inserts.emplace_back(insertOp.getValue(), JIT::Storage::Constant(reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(container.getAddress()) + off), valueSize));
                    });
                })
                .Case([&](mlir::LLVM::AddressOfOp addrOp) {
                    foreach_storage(addrOp, [](auto) {});
                })
                .Case([&](mlir::LLVM::GEPOp gepOp) {
                    for (uint32_t i = 1; i < gepOp.getNumOperands(); i++) {
                        ASSURE(getBitSize(gepOp.getOperand(i).getType()), <=, 64);
                        if (storedConstants.count(gepOp.getOperand(i)) == 0) {
                            auto val = std::make_unique<uint64_t[]>(1);
                            auto* ptr = val.get();
                            jit.m_aliveConstants.emplace_back(std::move(val));
                            storedConstants.emplace(gepOp.getOperand(i), JIT::Storage::Constant(ptr, TypeSize(64)));
                        }
                    }
                    foreach_storage(gepOp, [](auto) {});
                })
                .Case([&](mlir::LLVM::ZExtOp zextOp) {
                    foreach_storage(zextOp, [&](const auto storage) {
                        inserts.emplace_back(zextOp.getArg(), JIT::Storage::Constant(storage.getAddress(), TypeSize::FromType(zextOp.getArg().getType())));
                    });
                })
                .Case([&](mlir::LLVM::IntToPtrOp itopOp) {
                    foreach_storage(itopOp, [&](const auto storage) {
                        inserts.emplace_back(itopOp.getArg(), storage);
                    });
                })
                .Case([&](mlir::LLVM::PtrToIntOp ptoiOp) {
                    foreach_storage(ptoiOp, [&](const auto storage) {
                        inserts.emplace_back(ptoiOp.getArg(), storage);
                    });
                })
                .Case([&](mlir::LLVM::ReturnOp returnOp) {
                    const auto size = TypeSize::FromType(returnOp.getOperand(0).getType());
                    auto val = std::make_unique<uint64_t[]>(size.slots());
                    auto* ptr = val.get();
                    jit.m_aliveConstants.emplace_back(std::move(val));
                    storedConstants.emplace(returnOp.getOperand(0), JIT::Storage::Constant(ptr, size));
                    jit.registerSymbol(op.getSymName(), reinterpret_cast<char*>(ptr));
                })
                .Default([]([[maybe_unused]] mlir::Operation* anyOp) {
                    UNREACHABLE("can not const propagate " << *anyOp);
                });
        }

        for (auto& o : op.getBodyRegion().getOps()) {
            llvm::TypeSwitch<mlir::Operation*, void>(&o)
                .Case([&](mlir::LLVM::NullOp nullOp) {
                    foreach_storage(nullOp, [](const auto storage) {
                        *storage.getAddress() = 0;
                    });
                })
                .Case([&](mlir::LLVM::ConstantOp constOp) {
                    auto constant = fillConstantFromAttr(jit, constOp.getValue(), constOp.getResult().getType());
                    foreach_storage(constOp, [&](const auto storage) {
                        ASSURE(constant.second, ==, storage.size());
                        memcpy(storage.getAddress(), constant.first.get(), constant.second.bytes());
                    });
                })
                .Case([&](mlir::arith::ConstantOp constOp) {
                    auto constant = fillConstantFromAttr(jit, constOp.getValue(), constOp.getResult().getType());
                    foreach_storage(constOp, [&](const auto storage) {
                        ASSURE(constant.second, ==, storage.size());
                        memcpy(storage.getAddress(), constant.first.get(), constant.second.bytes());
                    });
                })
                .Case([&](mlir::LLVM::AddressOfOp addrOp) {
                    const auto symName = addrOp.getGlobalName();
                    foreach_storage(addrOp, [&](const auto storage) {
                        *storage.getAddress() = 0;
                        jit.m_symbols.insertFixupLocation(symName.str(), {reinterpret_cast<char*>(storage.getAddress()), true});
                    });
                })
                .Case([&](mlir::LLVM::GEPOp gepOp) {
                    ASSURE_TRUE(llvm::isa<mlir::LLVM::AddressOfOp>(*gepOp.getBase().getDefiningOp()));
                    std::vector<int64_t> indices;
                    for (auto dynamic : gepOp.getDynamicIndices()) {
                        ASSURE(storedConstants.count(dynamic), >=, 1);
                        ASSURE(storedConstants.find(dynamic)->second.size().slots(), ==, 1);
                        indices.push_back(storedConstants.find(dynamic)->second.getAddress()[0]);
                    }
                    ASSURE_TRUE(std::all_of(gepOp.getRawConstantIndices().begin(), gepOp.getRawConstantIndices().end(), [](const int32_t ind) {
                        return ind == INT32_MIN;
                    }));
                    mlir::Type containerT;
                    if (gepOp.getElemType().has_value() && gepOp.getElemType()->operator bool()) {
                        containerT = mlir::LLVM::LLVMArrayType::get(
                            gepOp.getContext(),
                            *gepOp.getElemType(),
                            indices.front() + 1);
                    } else {
                        containerT = gepOp.getOperand(0).getType();
                    }
                    const auto offset = accumulateOffsets(containerT, indices.cbegin(), indices.cend());
                    const auto symNameBase = llvm::cast<mlir::LLVM::AddressOfOp>(*gepOp.getBase().getDefiningOp()).getGlobalName();
                    foreach_storage(gepOp, [&](const auto storage) {
                        *storage.getAddress() = offset;
                        jit.m_symbols.insertFixupLocation(symNameBase.str(), {reinterpret_cast<char*>(storage.getAddress()), true});
                    });
                })
                .Default([](mlir::Operation*) {});
        }
    } else if (op.getValue().has_value()) {
        // derive value from attribute
        auto [val, size] = fillConstantFromAttr(
            jit,
            op.getValue().value(),
            op.getType());
        ASSURE_TRUE(val.get() != nullptr);
        auto* ptr = reinterpret_cast<char*>(val.get());
        jit.m_aliveConstants.emplace_back(std::move(val));
        DEBUG("Register symbol: " << op.getSymName() << " with size: " << size);
        jit.registerSymbol(op.getSymName(), ptr);
    } else {
        // if we have neither a value nor a computing region, the symbol must be present already...
        void* ptr = dlsym(RTLD_DEFAULT, op.getSymName().str().c_str());
        ASSURE_TRUE(ptr != nullptr, << " missing: " << op.getSymName());
        DEBUG("Register host binary symbol: " << op.getSymName());
        jit.registerSymbol(op.getSymName(), ptr);
    }
}

void CustomPattern<mlir::LLVM::AllocaOp>::Codegen(
    mlir::LLVM::AllocaOp& op,
    JIT& jit,
    uint16_t,
    uint16_t cgDepth)
{
    if (!llvm::isa<mlir::LLVM::ConstantOp, mlir::arith::ConstantOp>(*op->getOperand(0).getDefiningOp())) {
        // var size stack allocation ...
        // we just do an allocation with malloc and basically never free it for now...
        std::vector<uint64_t> args;
        jit.prepareOpCall(op.operator mlir::Operation*(), cgDepth, args, false);
        ASSURE_TRUE(args.empty());
        // TODO: assure the correct multiplicity (if allocated elements are larger than one slot)

        const auto templateId = JIT::Materializer<Malloc>().templateId;
        const auto& templateInfo = jit.m_library.getTemplate(templateId);
        const auto& templateCode = std::get<2>(templateInfo);
        const auto size = templateCode.size();
        LOG_JIT(cgDepth, size, Malloc::templateName(), templateId);
        char* before = jit.memory(cgDepth, size);
        templateCode.instantiate(before, 0);
        templateCode.patch(
            {before, jit.memory(cgDepth)},
            {},
            [&](void* originalAddr) {
                const auto addr = jit.memory(cgDepth + 1);
                JIT::Materializer<ExternalThunk>()(jit, cgDepth + 1, reinterpret_cast<uint64_t>(originalAddr));
                return addr;
            },
            {},
            {});
        auto newStorage = jit.nextSlot<JIT::Storage::STACK>(TypeSize(64));
        jit.moveTo(cgDepth, JIT::Storage::Register(1), newStorage);
        jit.setStorage(op.getResult(), newStorage);
        jit.applyClobberMask(CLOBBER_ALL);
        return;
    }
    // we only support constant stack allocations
    auto T = op.getElemType().value_or(op.getType().getElementType());
    size_t byteSize = (getBitSize(T) + 7) / 8;
    if (op->getNumOperands() >= 1) {
        ASSURE(op->getNumOperands(), ==, 1);
        ASSURE_TRUE(op->getOperand(0).getDefiningOp() != nullptr);
        ASSURE_TRUE(op->getOperand(0).getDefiningOp()->hasAttrOfType<mlir::IntegerAttr>("value"));
        auto value = op->getOperand(0).getDefiningOp()->getAttrOfType<mlir::IntegerAttr>("value");
        byteSize *= value.getValue().getZExtValue();
    }
    auto stackSlot = jit.nextSlot<JIT::Storage::STACK>(TypeSize(byteSize * 8, op.getAlignment().value_or(0)));
    ASSURE_TRUE(llvm::isa<mlir::FileLineColLoc>(op->getLoc()));
#if defined(__x86_64__)
    [[maybe_unused]] const int codeSize = 7;
#elif defined(__aarch64__)
    [[maybe_unused]] const int codeSize = (stackSlot.getOffset() * 8) > 4095 ? 8 : 4;
#endif
    LOG_JIT(
        cgDepth,
        codeSize,
        "Alloca",
        static_cast<uint32_t>(-1),
        llvm::cast<mlir::FileLineColLoc>(op->getLoc()).getLine());
    ASM(STACK_SLOT_ADDRESS, static_cast<uint32_t>(stackSlot.getOffset() * 8));

    auto newStorage = jit.nextSlot<JIT::Storage::STACK>(TypeSize(64));
    jit.moveTo(cgDepth, JIT::Storage::Register(1), newStorage);
#ifdef TMP_REGISTER_COPY
    auto r = op.getResult();
    // scatter somewhere into a scratch register
    auto tmp = jit.tmpRegister(r);
    if (tmp != 0) {
        ASSURE_TRUE(newStorage.isStack());
        newStorage = JIT::Storage::Stack(newStorage.getOffset(), TypeSize(64), tmp);
        jit.moveTo(cgDepth, JIT::Storage::Register(1), JIT::Storage::Register(tmp));
        jit.set(tmp, r);
    }
#endif
    jit.setStorage(op.getResult(), newStorage);
    // clobber rbp
    jit.applyClobberMask(0b0000000001);
}

void CustomPattern<mlir::LLVM::AddressOfOp>::Codegen(
    mlir::LLVM::AddressOfOp& op,
    JIT& jit,
    uint16_t,
    uint16_t cgDepth)
{
    const auto symName = op.getGlobalName();
    void* symbolAddress = jit.m_symbols.getOrNull(symName);
    auto newSlot = jit.nextSlot<JIT::Storage::STACK>(TypeSize(64));
    if (symbolAddress == nullptr) {
#if defined(__x86_64__)
        jit.m_symbols.insertFixupLocation(symName.str(), jit.memory(cgDepth) + sizeof(LEA_SCRATCH));
        [[maybe_unused]] const int codeSize = sizeof(LEA_SCRATCH) + sizeof(MOV_SCRATCH_TO_STACK) + 8;
#elif defined(__aarch64__)
        [[maybe_unused]] const int codeSize = 12;
#endif
        ASSURE_TRUE(llvm::isa<mlir::FileLineColLoc>(op->getLoc()));
        LOG_JIT(
            cgDepth,
            codeSize,
            "AddressOf",
            static_cast<uint32_t>(-1),
            llvm::cast<mlir::FileLineColLoc>(op->getLoc()).getLine());
#if defined(__x86_64__)
        ASM(LEA_SCRATCH, static_cast<int32_t>(-4));
#elif defined(__aarch64__)
        const uint32_t nop = 0xd503201f;
        jit.insert(cgDepth, 4, &nop);
        ASM(LEA_SCRATCH);
        jit.m_symbols.insertFixupLocation(symName.str(), jit.memory(cgDepth) - 4);
#endif
        ASM(MOV_SCRATCH_TO_STACK, static_cast<uint32_t>(newSlot.getOffset() * 8));
        jit.applyClobberMask(CLOBBER_SCRATCH);
    } else {
        JIT::Materializer<MoveConstantToStack>()(jit, cgDepth, static_cast<uint64_t>(newSlot.getOffset()) * 8, reinterpret_cast<uint64_t>(symbolAddress));
    }
    jit.setStorage(op.getResult(), newSlot);
}

void CustomPattern<mlir::LLVM::InsertValueOp>::Codegen(
    mlir::LLVM::InsertValueOp& op,
    JIT& jit,
    uint16_t,
    uint16_t cgDepth)
{
    const auto container = op.getContainer();
    if (container.use_empty())
        return;

    auto storedContainer = jit.getStorage(container);
    const auto storedValue = jit.getStorage(op.getValue());
    const auto indices = op.getPosition().vec();
    auto off = accumulateOffsets(container.getType(), indices.cbegin(), indices.cend());
    auto byteSize = storedValue.size().bytes();

    if (container.hasOneUse() && storedContainer.isConstant() && storedValue.isConstant()) {
        // we can do some kind of constant propagation is this case
        memcpy(reinterpret_cast<uint8_t*>(storedContainer.getAddress()),
            reinterpret_cast<uint8_t*>(storedValue.getAddress()), byteSize);
        jit.setStorage(op.getResult(), storedContainer);
        return;
    }
    if (!container.hasOneUse() || storedContainer.isConstant()) {
        // in case of multiple uses for that container, we can not do the insertion in-place but have to
        // copy the container -> copy it to new location and do insertion in-place there
        // As an optimization the "last" use could be in-place again
        // same applies for a constant container, as it is no longer constant when modified during runtime
        // therefore, copy it to stack first and afterwards operate in-place
        const auto copiedContainer = jit.nextSlot<JIT::Storage::STACK>(storedContainer.size());
        jit.moveTo(cgDepth, storedContainer, copiedContainer);
        storedContainer = copiedContainer;
    }
    ASSURE_TRUE(storedContainer.isStack());
    ASSURE_TRUE(storedValue.isStack() || storedValue.isConstant());
    uint32_t i = 0;
    [[maybe_unused]] void* addrBefore = jit.memory(cgDepth);
    const auto moveByte = [&]() {
        // move byte by byte
        if (storedValue.isConstant()) {
            ASM(MOV_CONST_TO_SCRATCH_BYTE,
                reinterpret_cast<uint8_t*>(storedValue.getAddress())[i]);
        } else {
            ASM(MOV_STACK_TO_SCRATCH_ZX,
                static_cast<uint32_t>(storedValue.getOffset() * 8 + i));
        }
        ASM(MOV_SCRATCH_TO_STACK_BYTE,
            static_cast<uint32_t>(storedContainer.getOffset() * 8 + off));
    };
    while (off % 8 != 0 && byteSize > 0) {
        moveByte();
        byteSize -= 1;
        off += 1;
        i += 1;
    }
    while (byteSize >= 8) {
        ASSURE_TRUE(storedContainer.isStack());
        ASSURE(off % 8, ==, 0);
        const auto fittingSize = byteSize & ~0b111;
        jit.moveTo(cgDepth, storedValue, JIT::Storage::Stack(static_cast<uint32_t>(storedContainer.getOffset()) + off / 8, TypeSize(fittingSize * 8)));
        byteSize -= fittingSize;
        off += fittingSize;
        i += fittingSize;
    }
    while (byteSize > 0) {
        moveByte();
        byteSize -= 1;
        off += 1;
        i += 1;
    }
    LOG_JIT(
        addrBefore,
        reinterpret_cast<uint64_t>(jit.memory(cgDepth)) - reinterpret_cast<uint64_t>(addrBefore),
        op->getName().getStringRef().str(),
        static_cast<uint32_t>(-1),
        llvm::cast<mlir::FileLineColLoc>(op->getLoc()).getLine());
    jit.setStorage(op.getResult(), storedContainer);
    jit.applyClobberMask(CLOBBER_ALL);
}
#ifdef LESS_TEMPLATES
void CustomPattern<mlir::LLVM::CallOp>::Codegen(
    mlir::LLVM::CallOp& op,
    JIT& jit,
    uint16_t sourceDepth,
    uint16_t cgDepth)
{
    genericCallOp(op, jit, sourceDepth, cgDepth);
}
#endif
