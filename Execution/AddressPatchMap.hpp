#include "common.hpp"

#include <variant>

template <template <typename> typename Map, typename KeyT>
class AddressPatchMap
{
    using LocationT = std::pair<char*, bool>;
    using AddrT = void*;
    using ValueT = std::variant<std::vector<LocationT>, AddrT>;

public:
    AddressPatchMap() = default;
    ~AddressPatchMap()
    {
        assureAllSymbolsResolved();
    }

    void insertFixupLocation(KeyT key, char* location)
    {
        insertFixupLocation(key, {location, false});
    }

    void insertFixupLocation(KeyT key, LocationT location)
    {
        auto result = m_map.try_emplace(key, std::vector<LocationT>{location});
        if (!result.second) {
            auto& elem = result.first->second;
            if (std::holds_alternative<AddrT>(elem)) {
                writeLocation(location, std::get<AddrT>(elem));
            } else {
#ifdef __aarch64__
                ASSURE_TRUE(location.second
                    || location.first[3] == (char)0b00010000
                    || location.first[3] == (char)0b00010100
                    || location.first[3] == (char)0b10010100);
#endif
                std::get<std::vector<LocationT>>(elem).push_back(location);
            }
        }
    }

    void ignore(KeyT key)
    {
        m_map.erase(key);
    }

    void insertFixupAddress(KeyT key, AddrT addr)
    {
        auto result = m_map.try_emplace(key, addr);
        if (!result.second) {
            auto& elem = result.first->second;
            ASSURE_TRUE(std::holds_alternative<std::vector<LocationT>>(elem), << key);
            for (LocationT loc : std::get<std::vector<LocationT>>(elem)) {
                writeLocation(loc, addr);
            }
            elem.template emplace<AddrT>(addr);
        }
    }

    void* getOrNull(KeyT key) const
    {
        auto it = m_map.find(key);
        if (it != m_map.end() && std::holds_alternative<AddrT>(it->second)) {
            return std::get<AddrT>(it->second);
        } else {
            return nullptr;
        }
    }

    void assureAllSymbolsResolved() const
    {
#ifdef NORELEASE
        bool anyUninit = false;
        for (const auto& [key, value] : m_map) {
            if (!std::holds_alternative<AddrT>(value)) {
                INFO("failed resolving: " << key);
                anyUninit = true;
            }
        }
        ASSURE_FALSE(anyUninit, << "failed resolving");
#endif
    }

private:
    static void writeLocation(LocationT location, AddrT addr)
    {
#if defined(__x86_64__)
        const auto addend = UnalignedRead<int32_t>(location.first);
        if (location.second) {
            UnalignedWrite(location.first, reinterpret_cast<uint64_t>(addr) + addend);
            return;
        }
        const int64_t relAddr = reinterpret_cast<int64_t>(addr) - (reinterpret_cast<int64_t>(location.first) - addend);
        if (relAddr == 0) {
            ASSURE(location.first[-1], ==, (char)0xe9);
            // this mainly targets the cases when we split up the code using -basic-block-sections and now stick it back together
            // in those cases jumps with a distance of zero might occur -> make them a nop
            location.first[-1] = (char)0x0f;
            location.first[0] = (char)0x1f;
            location.first[1] = (char)0x44;
            location.first[2] = (char)0x00;
            location.first[3] = (char)0x00;
            return;
        }
        ASSURE(relAddr, >=, INT32_MIN);
        ASSURE(relAddr, <=, INT32_MAX);
        UnalignedWrite(location.first, static_cast<int32_t>(relAddr));
#elif defined(__aarch64__)
        if (location.second) {
            UnalignedWrite(location.first, reinterpret_cast<uint64_t>(addr));
            return;
        }
        switch (location.first[3]) {
            // adr
            case (char)0b00010000: {
                ASSURE_FALSE(location.second);
                const int64_t relAddr = reinterpret_cast<int64_t>(addr) - (reinterpret_cast<int64_t>(location.first));
                if (relAddr >= (-(1 << 20)) && relAddr <= (1 << 20)) {
                    ASSURE(relAddr & 0b11, ==, 0);
                    *reinterpret_cast<uint32_t*>(location.first) |= ((static_cast<uint32_t>(relAddr) >> 2) & ((1 << 19) - 1)) << 5;
                } else {
                    // if we ran out of range, we know there was a second instruction for this relocation that we replaced with a nop
                    // therefore go backwards, find it and reuse it again to extend the range here...
                    uint32_t* secondLocation = reinterpret_cast<uint32_t*>(location.first);
                    uint32_t* nopslot = reinterpret_cast<uint32_t*>(location.first);
                    while (*nopslot != 0xd503201f)
                        nopslot--;
                    // adrp reg, #lo
                    // add reg, reg, #imm
                    const auto nopSlotPage = reinterpret_cast<int64_t>(nopslot) & ~0b111111111111;
                    const auto relAddrToNopSlot = reinterpret_cast<int64_t>(addr) - (nopSlotPage);
                    ASSURE(relAddrToNopSlot, >=, INT32_MIN);
                    ASSURE(relAddrToNopSlot, <=, INT32_MAX);
                    const auto pageRel = static_cast<int32_t>(relAddrToNopSlot) >> 12;
                    *nopslot = 0x90000000
                        | (*secondLocation & 0b11111)
                        | (static_cast<uint32_t>(pageRel & 0b11) << 29)
                        | ((static_cast<uint32_t>(pageRel >> 2) & ((1u << 19) - 1)) << 5);
                    const auto additionalDistance = reinterpret_cast<int64_t>(addr) - (nopSlotPage + pageRel * (1 << 12));
                    ASSURE(additionalDistance, >=, 0);
                    ASSURE(additionalDistance, <=, (1 << 12) - 1);
                    *secondLocation = 0x91000000
                        | (*secondLocation & 0b11111)
                        | ((*secondLocation & 0b11111) << 5)
                        | (static_cast<uint32_t>(additionalDistance) << 10);
                }
            } break;
            case (char)0b00010100: // b
            case (char)0b10010100: // bl
            {
                const int64_t relAddr = reinterpret_cast<int64_t>(addr) - (reinterpret_cast<int64_t>(location.first));
                ASSURE(relAddr, >=, (-(1 << 27)));
                ASSURE(relAddr, <=, (1 << 27));
                *reinterpret_cast<uint32_t*>(location.first) |= (static_cast<int32_t>(relAddr) >> 2) & ((1 << 26) - 1);
            } break;
            default:
                UNREACHABLE(<< static_cast<uint32_t>(location.first[3]));
        }
#endif
    }

    Map<ValueT> m_map;
};
