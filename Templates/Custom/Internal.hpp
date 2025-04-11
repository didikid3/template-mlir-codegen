#define REGISTER_INTERNAL(name, count, ...)               \
    struct name                                           \
    {                                                     \
        static uint32_t id(__VA_ARGS__);                  \
        static std::string templateName()                 \
        {                                                 \
            return std::string(#name);                    \
        }                                                 \
        static constexpr uint32_t TEMPLATE_COUNT = count; \
    };                                                    \
    template <>                                           \
    struct CustomPattern<name>                            \
    {                                                     \
        static void GenerateTemplate(                     \
            TemplateBuilder& builder,                     \
            uint32_t id,                                  \
            uint32_t nr);                                 \
        static constexpr uint32_t TEMPLATE_COUNT = count; \
    };

REGISTER_INTERNAL(MoveStackToRegister, OPERAND_REGISTERS, uint32_t);
REGISTER_INTERNAL(MoveBetweenRegisters, CACHE_REGISTERS * CACHE_REGISTERS, uint32_t, uint32_t);
REGISTER_INTERNAL(MoveConstantToStack, 1);
REGISTER_INTERNAL(MoveConstantToRegister, OPERAND_REGISTERS, uint32_t);
REGISTER_INTERNAL(CopyOnStack, 1);
REGISTER_INTERNAL(RegionTerminator, 1);
REGISTER_INTERNAL(MoveRegisterToStack, OPERAND_REGISTERS, uint32_t);
REGISTER_INTERNAL(ExternalThunk, 1);
REGISTER_INTERNAL(Br, 1);
REGISTER_INTERNAL(Malloc, 1);

#undef REGISTER_INTERNAL
