#pragma once

#include "Execution/StackExecutor.hpp"
#include "Preparation/CustomPass.hpp"
#include "Templates/PredefinedPatterns.hpp"

#include <iostream>
#include <fstream>
#include <mutex>

class RegisterLogger {
public:
    static RegisterLogger& instance(){
        static RegisterLogger instance;
        return instance;
    }

    void log(const std::string& message){
        std::lock_guard<std::mutex> lock(m_mutex);
        std::ofstream logFile("register_cache.log", std::ios::out | std::ios::app);
        if (logFile.is_open()) {
            logFile << message << std::endl;
        }
    }

    ~RegisterLogger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }
private:
    RegisterLogger(){}
    std::ofstream logFile;
    std::mutex m_mutex;
};

template <typename Impl>
class RegisterExecutor : public StackExecutor<Impl>
{
public:
    using Base = StackExecutor<Impl>;
    using Base::Base;

    using Storage = typename Base::Storage;

    void reset()
    {
        Base::reset();
        m_validRegisters.reset();
        evictionIndex = 2;
        maxEvictionIndex = CACHE_REGISTERS - 1;
    }

    void clearSlot(Storage storage)
    {
        if (storage.isStack()) {
#ifdef TMP_REGISTER_COPY
            if (storage.size().slots() <= 1 && storage.tmpRegister()) {
                m_validRegisters.reset(storage.tmpRegister() - 1);
            }
#endif
            Base::clearSlot(Storage::Stack(storage.getOffset(), storage.size()));
        }
    }

    bool isValid([[maybe_unused]] uint8_t addr, [[maybe_unused]] mlir::Value value)
    {
#ifdef TMP_REGISTER_COPY
        // return addr > 0 && m_validRegisters.test(addr - 1) && m_values[addr - 1] == value;
    if (addr > 0 && m_validRegisters.test(addr - 1) && m_values[addr - 1] == value) {
        m_usage[addr - 1] = ++m_timestamp; // Recent use update
        registerFrequencies[addr]++; // update frequency

        std::stringstream msg;
        msg << " A " << std::to_string(addr-1) << " TRUE";
        RegisterLogger::instance().log(msg.str());

        return true;
    }
        std::stringstream msg;
        msg << " A " << std::to_string(addr-1) << " FALSE";
        RegisterLogger::instance().log(msg.str());

        return false;
#else
        return false;
#endif
    }

    void set(uint8_t reg, mlir::Value value)
    {
        ASSURE(reg, >, 0);
        ASSURE(reg, <=, m_validRegisters.size());
        ASSURE_FALSE(m_validRegisters[reg - 1]);
        m_validRegisters.set(reg - 1);
        m_values[reg - 1] = value;
        m_usage[reg - 1] = ++m_timestamp; // Update usage
        registerFrequencies[reg]++; // update frequency

        std::stringstream msg;
        msg << " E " << std::to_string(reg-1);
        RegisterLogger::instance().log(msg.str());
    }

    uint8_t tmpRegister(mlir::Value)
    {
#if EVICTION_STRATEGY == NO_EVICT
        // in range [1, CACHE_REGISTERS] is valid (zero means not available)
        const uint32_t v = static_cast<uint32_t>(~m_validRegisters.to_ulong())
            & static_cast<uint32_t>(CLOBBER_ALL.to_ulong());
        if (v) {
            const auto lz = 32 - __builtin_clz(v);
            return lz > 2 ? lz : 0;
        }
        return 0;
#elif EVICTION_STRATEGY == EVICTION_ROUND_ROBIN
        // in range [1, CACHE_REGISTERS] is valid (zero means not available)
        const uint32_t v = static_cast<uint32_t>(~m_validRegisters.to_ulong())
            & static_cast<uint32_t>(CLOBBER_ALL.to_ulong());
        if (v) {
            const auto lz = 32 - __builtin_clz(v);
            if (lz > 2)
                return lz;
        }
        evictionIndex = (evictionIndex == maxEvictionIndex) ? 2 : evictionIndex + 1;
        ASSURE(evictionIndex, <=, maxEvictionIndex);
        return evictionIndex + 1;
#elif EVICTION_STRATEGY == EVICTION_LRU
        // in range [1, CACHE_REGISTERS] is valid (zero means not available)
        const uint32_t v = static_cast<uint32_t>(~m_validRegisters.to_ulong())
            & static_cast<uint32_t>(CLOBBER_ALL.to_ulong());
        if (v) {
            const auto lz = 32 - __builtin_clz(v);
            return lz;
        }
        // No free registers available, select LRU for eviction
        uint64_t minUsage = UINT64_MAX;
        uint8_t lruIndex = 2; // start from first usable register
        for (uint8_t i = 2; i < CACHE_REGISTERS; ++i) {
            if (m_usage[i] < minUsage) {
                minUsage = m_usage[i];
                lruIndex = i;
            }
        }
        ASSURE(lruIndex, <=, maxEvictionIndex);
        return lruIndex + 1;
#elif EVICTION_STRATEGY == EVICTION_LFU
        // in range [1, CACHE_REGISTERS] is valid (zero means not available)
        const uint32_t v = static_cast<uint32_t>(~m_validRegisters.to_ulong())
            & static_cast<uint32_t>(CLOBBER_ALL.to_ulong());
        if (v) {
            const auto lz = 32 - __builtin_clz(v);
            return lz;
        }
        // No free registers available, select LFU for eviction
        // Eviction based on frequency of use
        uint64_t minFreq = UINT64_MAX;
        uint8_t regToEvict = 0; 
        for(uint8_t i = 3; i < CACHE_REGISTERS; ++i) {
            if(m_validRegisters.test(i-1)) {
                if(registerFrequencies[i] < minFreq) {
                    minFreq = registerFrequencies[i];
                    regToEvict = i;
                }
            }
        }
        ASSURE(regToEvict, >, 2);   // ensure you're not evicting reserved registers (0–2)
        ASSURE(regToEvict, <=, maxEvictionIndex); 
        registerFrequencies[regToEvict] = 1; // Reset frequency of the evicted register
        return regToEvict;
#else
        return 0;
#endif
    }

    void applyClobberMask([[maybe_unused]] const ClobberMask& clobbers)
    {
#ifdef TMP_REGISTER_COPY
        m_validRegisters &= ~clobbers;
#endif
    }

#ifndef RELEASE
    uint32_t savedStackLoads = 0;
    uint32_t stackLoads = 0;
#endif

private:
    ClobberMask m_validRegisters{};
    std::array<mlir::Value, CACHE_REGISTERS> m_values{};
    std::array<uint64_t, CACHE_REGISTERS> m_usage{}; // Array to track the timestamps
    std::array<uint32_t, CACHE_REGISTERS + 1> registerFrequencies{}; // index 0 unused
    uint64_t m_timestamp = 0; 
    mutable uint32_t evictionIndex = 2;
    uint32_t maxEvictionIndex = CACHE_REGISTERS - 1;
};
