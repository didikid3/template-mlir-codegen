#pragma once

#include "Execution/Executor.hpp"

template <typename Impl>
class StackExecutor : public Executor<Impl>
{
public:
    using Base = Executor<Impl>;
    using Base::Base;

    using Storage = typename Base::Storage;

    template <uint8_t StorageT>
    Storage nextSlot(TypeSize size)
    {
        static_assert(StorageT == Storage::STACK);
        const auto slotSize = size.slots();
        if (size.align() > 8) {
            // it regularly happens, that we load a vector using vector instructions from stack
            // those require a certain alignment (8, 16, 32 up to 64 byte aligned)
            // for the worst case, we align everything to MAX_ALIGN byte
            static_assert(MAX_ALIGN % 8 == 0);
            const uint32_t MAX_SLOT_ALIGN = MAX_ALIGN / 8;
            m_currentStackOffset = (m_currentStackOffset + MAX_SLOT_ALIGN - 1) & -MAX_SLOT_ALIGN;
            ASSURE(m_currentStackOffset % 8, ==, 0);
        }
        ASSURE(m_currentStackOffset + slotSize, <, MAX_STACK_SLOTS);

        if (m_usedSlots.size() < m_currentStackOffset + slotSize) {
            m_usedSlots.resize(std::max(static_cast<uint32_t>(2 * m_usedSlots.size()), m_currentStackOffset + slotSize));
        }
        for (unsigned i = 0; i < slotSize; i++) {
            m_usedSlots[m_currentStackOffset + i] = true;
        }
        m_currentStackOffset += slotSize;
        m_maxStackSpace = std::max(m_maxStackSpace, m_currentStackOffset);

        return Storage::Stack(m_currentStackOffset - slotSize, size);
    }

    uint32_t stackOffset()
    {
        return m_currentStackOffset;
    }

    uint32_t m_maxStackSpace = 0;

protected:
    void reset()
    {
        m_usedSlots = std::vector<bool>(128, false);
        m_currentStackOffset = 0;
        m_currentStackFrame = 0;
        m_maxStackSpace = 0;
    }

    void clearSlot(const Storage& storage)
    {
        ASSURE_TRUE(storage.isStack());
        const auto offset = storage.getOffset();
        for (size_t i = offset; i < offset + storage.size().slots(); i++)
            m_usedSlots[i] = false;

        while ((m_currentStackOffset > m_currentStackFrame) && !m_usedSlots[m_currentStackOffset - 1]) {
            m_currentStackOffset--;
        }
    }

    class StackFrame
    {
    public:
        StackFrame(StackExecutor& exec, const StackMap& stackmap, mlir::Region& region, Storage storage)
            : m_exec(exec)
            , m_stackOffsetReset(exec.m_currentStackOffset)
            , m_stackFrameReset(exec.m_currentStackFrame)
        {
            exec.m_currentStackFrame = m_stackOffsetReset;
            const auto args = region.getArguments();
            ASSURE(args.size(), ==, stackmap.size());
            uint32_t totalSlotSize = 0;
            for (size_t i = 0; i < args.size(); i++) {
                ASSURE(storage.isStack(), ||, storage.isRegister());
                const auto size = stackmap[i];
                totalSlotSize += size.slots();
                m_exec.setStorage(args[i], Storage::Stack(storage.getOffset(), size));
                storage = Storage::Stack(storage.getOffset() + size.slots(), TypeSize(0));
            }
            // to "reserve" the actual memory from the stack, allocate just a single big slot
            m_exec.nextSlot<Storage::STACK>(TypeSize(totalSlotSize * 64));
        }

        ~StackFrame()
        {
            m_exec.m_currentStackFrame = m_stackFrameReset;
            m_exec.m_currentStackOffset = m_stackOffsetReset;
            m_exec.m_usedSlots[m_exec.m_currentStackOffset] = false;
        }

    private:
        StackExecutor& m_exec;
        const uint32_t m_stackOffsetReset;
        const uint32_t m_stackFrameReset;
    };

private:
    uint32_t m_currentStackOffset = 0;
    uint32_t m_currentStackFrame = 0;
    std::vector<bool> m_usedSlots = std::vector<bool>(128, false);
};
