#pragma once
#include "Preparation/CustomPass.hpp"
#include "common.hpp"

#include "Execution/MemInterface.hpp"

#include "Preparation/Abstraction.hpp"
#include "Templates/Template.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#ifndef LINGODB_LLVM
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#endif
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#ifdef VTUNE_SUPPORT
#include "ittnotify.h"
#endif

#include <bitset>
#include <chrono>
#include <fstream>

#include <array>

struct TimedResult
{
    const uint64_t result;

    // times in microseconds
    const long long prepareTime;
    const long long executionTime;
    const long long loweringTime;
    const std::string name;
};

inline std::ostream& operator<<(std::ostream& os, const TimedResult& result)
{
    os << "\n";
    os << "=====================|" << result.name << "|=====================\n";
    if (result.loweringTime > 0)
        os << "Lowering    Time : " << result.loweringTime << " μs \n";
    os << "Preparation Time : " << result.prepareTime << " μs \n";
    os << "Execution   Time : " << result.executionTime << " μs \n";
    os << std::string(result.name.size(), '=') << "============================================\n";
    return os;
}

#ifdef FORTRAN
static TimedResult* printResult = nullptr;
static decltype(std::chrono::steady_clock::now()) beforeRun;
inline void atexit_print_handler()
{
    auto after = std::chrono::steady_clock::now();
    if (TIME_FLAG) {
        ASSURE_TRUE(printResult != nullptr);
        auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(after - beforeRun).count();
        auto file = std::ofstream(printResult->name + "-time.txt");
        file << TimedResult{0, printResult->prepareTime, executionTime, printResult->loweringTime, printResult->name};
        file.close();
    }
}
#endif

template <typename Impl>
class Executor
{
public:
    Executor(const std::string& name)
        : m_name(name)
    {
    }

    template <typename T>
    void prepare(T* mod)
    {
#ifdef VTUNE_SUPPORT
        __itt_task_begin(domain, __itt_null, __itt_null, prepareTask);
#endif
        auto before = std::chrono::steady_clock::now();
        // memory mapping also counts toward the preparation time
        static_cast<Impl*>(this)->prepareImpl(mod);
        auto after = std::chrono::steady_clock::now();
        prepareTime = std::chrono::duration_cast<std::chrono::microseconds>(after - before).count();
#ifdef VTUNE_SUPPORT
        __itt_task_end(domain);
#endif
    }

    uint64_t run(mlir::ModuleOp* mod)
    {
        if (COMPILE_ONLY)
            return 0;
#ifdef VTUNE_SUPPORT
        __itt_task_begin(domain, __itt_null, __itt_null, executeTask);
#endif
        auto before = std::chrono::steady_clock::now();
#ifdef FORTRAN
        beforeRun = before;
#endif
        auto result = static_cast<Impl*>(this)->runImpl(mod);
        auto after = std::chrono::steady_clock::now();
        executionTime = std::chrono::duration_cast<std::chrono::microseconds>(after - before).count();
#ifdef VTUNE_SUPPORT
        __itt_task_end(domain);
#endif
        return result;
    }

    template <typename T>
    T* lower(mlir::ModuleOp* mod)
    {
        if constexpr (std::is_same_v<T, llvm::Module>) {
            auto before = std::chrono::steady_clock::now();
            m_llvmContext = std::make_unique<llvm::LLVMContext>();
            Abstraction::quickLowering(*mod->operator mlir::Operation*());
#ifndef LINGODB_LLVM
            mlir::registerBuiltinDialectTranslation(*mod->getContext());
#endif
            mlir::registerLLVMDialectTranslation(*mod->getContext());
            m_llvmModule = mlir::translateModuleToLLVMIR(*mod, *m_llvmContext);
            ASSURE_TRUE(m_llvmModule);
            auto after = std::chrono::steady_clock::now();
            loweringTime = std::chrono::duration_cast<std::chrono::microseconds>(after - before).count();
            return m_llvmModule.get();
        } else {
            loweringTime = 0;
            return mod;
        }
    }

    TimedResult operator()(mlir::ModuleOp* mod, const std::vector<uint64_t>& args)
    {
        RESET_TIMERS();
        constexpr bool REQUIRES_LOWERING = std::is_invocable_v<decltype(&Impl::prepareImpl), Impl&, llvm::Module*>;
        using ModT = std::conditional_t<REQUIRES_LOWERING, llvm::Module, mlir::ModuleOp>;

        INFO("Start Preparation of " << m_name << "...");

        if (m_argumentSizes.empty()) {
            m_argumentVector = args;
            m_argumentSizes = std::vector<uint8_t>(m_argumentVector.size(), 1);
        }

        prepare(lower<ModT>(mod));
#ifdef FORTRAN
        if (TIME_FLAG)
            printResult = new TimedResult{0, prepareTime, 0, loweringTime, m_name};
#endif
        INFO("Start Execution of " << m_name << "...");
        uint64_t result = 0;
        for (int i = 0; i < RUNS; i++)
            result = run(mod);

        INFO("Finished.");
        return TimedResult{result, prepareTime, executionTime, loweringTime, m_name};
    }

#ifdef VTUNE_SUPPORT
    __itt_domain* domain = __itt_domain_create("mlir-codegen");
    __itt_string_handle* prepareTask = __itt_string_handle_create("Preparation");
    __itt_string_handle* executeTask = __itt_string_handle_create("Execution");
#endif
    std::vector<uint64_t> m_argumentVector{};
    std::vector<uint8_t> m_argumentSizes{};

    class Storage
    {
    public:
        // use the least two bits for distinguishing between
        // stack (= 0), register (=1) and const (=2)
        // stack and register store in the following bits the offset
        // constant stores a uint64_t* ptr (which is 8-byte aligned and therefore lower bits are unused)

        static constexpr uint8_t STACK = 0;
        static constexpr uint8_t REGISTER = 1;
        static constexpr uint8_t CONSTANT = 2;

        bool isStack() const { return mask() == STACK; }
        bool isRegister() const { return mask() == REGISTER; }
        bool isConstant() const { return mask() == CONSTANT; }

        uint8_t mask() const
        {
            return m_kind;
        }

        bool same(const Storage& rhs) const
        {
            return rhs.m_kind == m_kind && rhs.m_storage == m_storage;
        }

        uint8_t tmpRegister() const
        {
            ASSURE_TRUE(isStack());
            ASSURE(m_size.slots(), ==, 1);
            return static_cast<uint8_t>(m_storage >> 59);
        }

        uint32_t getOffset() const
        {
            ASSURE_TRUE((isStack() || isRegister()));
            return static_cast<uint32_t>(m_storage);
        }
        uint64_t* getAddress() const
        {
            ASSURE_TRUE(isConstant());
            return reinterpret_cast<uint64_t*>(m_storage);
        }

        TypeSize size() const
        {
            return m_size;
        }

        /**
         * @brief Create storage for certain number of stack slots
         *
         * @param off stack offset
         * @param size value size
         * @param alternativeRegister register where the value is cached as well (0 indicates no cache register)
         * @return Storage
         */
        static Storage Stack(uint32_t off, TypeSize size, uint8_t alternativeRegister = 0)
        {
            ASSURE(alternativeRegister, <=, 31);
            // XXXXX0000000000000000000000000AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            // X : alternative register
            // A : offset
            ASSURE(static_cast<uint32_t>(alternativeRegister), <=, CACHE_REGISTERS);
            return Storage{(static_cast<uint64_t>(alternativeRegister) << 59) | static_cast<uint64_t>(off), size, STACK};
        }

        /**
         * @brief Create storage in continuous registers
         *
         * @param off register index greater than zero (0 corresponds to our value storage pointer)
         * @param size value size
         * @return Storage
         */
        static Storage Register(uint32_t off, TypeSize size = TypeSize(64))
        {
            ASSURE(off, >, 0);
            ASSURE(off, <=, CACHE_REGISTERS);
            return Storage{off, size, REGISTER};
        }

        /**
         * @brief Create storage of a compile-time constant value
         *
         * @param addr pointer to the actual stored value
         * @param size value size
         * @return Storage
         */
        static Storage Constant(uint64_t* addr, TypeSize size)
        {
            return Storage{reinterpret_cast<uint64_t>(addr), size, CONSTANT};
        }
        Storage() = default;

    private:
        Storage(uint64_t data, TypeSize size, uint8_t tag)
            : m_storage(data)
            , m_size(size)
            , m_kind(tag)
        {
        }

        uint64_t m_storage;
        TypeSize m_size;
        uint8_t m_kind;
    };

    const Storage& getOrCreateSlot(const mlir::Value& value)
    {
        MEASURE(ScopeTimer::valueStorage);
        auto result = m_varOffsets.try_emplace(value, Storage());
        if (result.second) {
            result.first->second = static_cast<Impl*>(this)->template nextSlot<Storage::STACK>(TypeSize::FromType(value.getType()));
        }
        return result.first->second;
    }

    const Storage& getOrCreateSlot(const mlir::Value& value, TypeSize size)
    {
        MEASURE(ScopeTimer::valueStorage);
        auto result = m_varOffsets.try_emplace(value, Storage());
        if (result.second) {
            result.first->second = static_cast<Impl*>(this)->template nextSlot<Storage::STACK>(size);
        }
        return result.first->second;
    }

    const Storage& getStorage(const mlir::Value& val) const
    {
        MEASURE(ScopeTimer::valueStorage);
        ASSURE(m_varOffsets.count(val), !=, 0,
            << " value " << val << " not declared before used(" << val.getLoc() << ")");
        return m_varOffsets.find(val)->second;
    }

    void setConstant(const mlir::Value& value, TypeSize size, uint64_t* data)
    {
        setStorage(value, Storage::Constant(data, size));
    }

    void setConstant(const mlir::Value& value, TypeSize size, std::unique_ptr<uint64_t[]>&& data)
    {
        uint64_t* ptr = data.get();
        m_aliveConstants.emplace_back(std::move(data));
        setStorage(value, Storage::Constant(ptr, size));
    }

    void setStorage(const mlir::Value& value, const Storage& storage, [[maybe_unused]] bool allowOverwrite = false)
    {
        MEASURE(ScopeTimer::valueStorage);
        ASSURE(allowOverwrite, ||, (m_varOffsets.count(value) == 0 || m_varOffsets.find(value)->second.same(storage)));
        m_varOffsets[value] = storage;
        ASSURE_TRUE(!storage.isStack() || (storage.getOffset() + storage.size().slots()) < MAX_STACK_SLOTS);
    }

    uint64_t invoke(void* function)
    {
        return invokeFromVector(function, m_argumentVector);
    }

    std::vector<std::unique_ptr<uint64_t[]>> m_aliveConstants;
    std::unique_ptr<llvm::LLVMContext> m_llvmContext;
    std::unique_ptr<llvm::Module> m_llvmModule;

private:
    llvm::DenseMap<mlir::Value, Storage> m_varOffsets;
    const std::string m_name;

    int64_t loweringTime;
    int64_t prepareTime;
    int64_t executionTime;
};
