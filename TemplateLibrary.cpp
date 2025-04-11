#include "TemplateLibrary.hpp"
#include "TemplateBuilder.hpp"
#include "Templates/PredefinedPatterns.hpp"
#include "common.hpp"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <numeric>

TemplateLibrary::TemplateLibrary(const std::string& name, mlir::MLIRContext& context)
    : CodeSectionProvider()
    , DataSectionProvider()
    , m_name(name)
    , m_context(context)
{
    std::filesystem::create_directory(name);
}

TemplateLibrary TemplateLibrary::Empty(const std::string& name, mlir::MLIRContext& context)
{
    std::filesystem::remove_all(name);
    TemplateLibrary lib{name, context};
    TemplateBuilder builder(lib, Abstraction::defaultLowering);
    CustomPatternGeneration::Generate(builder);
    return lib;
}
TemplateLibrary TemplateLibrary::FromFolder(const std::string& name, mlir::MLIRContext& context)
{
    TemplateLibrary lib{name, context};
    TemplateBuilder builder(lib, Abstraction::defaultLowering);
    CustomPatternGeneration::Generate(builder);

    if (std::filesystem::exists(name + "/.meta")) {
        lib.readMetadata();
    }
    return lib;
}

TemplateLibrary::IdType TemplateLibrary::insert(
    KeyType&& signature,
    ResultSizes&& returnSizes,
    StackMaps&& stackMaps,
    const mlir::ModuleOp& mod /*the stack args function*/)
{
    const auto id = m_templateStorage.size();
    ASSURE(m_signatureToId.count(signature), ==, 0);
    const auto isConst = signature.isConstexpr();
    m_signatureToId.try_emplace(std::move(signature), id);
    m_templateStorage.emplace_back(
        std::move(returnSizes),
        std::move(stackMaps),
        mod, isConst);
    return id;
}

void TemplateLibrary::addBinaryTemplate(
    IdType id,
    TEMPLATE_KIND kind,
    Template&& binTemplate)
{
    ASSURE(id, <, m_templateStorage.size());

    if (isConstant(id)) {
        return;
    }

    if (TEMPLATE_CONST_EVAL && kind == TEMPLATE_KIND::STACK_ARGS) {
        // const eval
        if (binTemplate.isReadyForExecution() && m_templateStorage[id].isConst) {
            const auto totalReturnSize = std::accumulate(
                m_templateStorage[id].returnSizes.begin(),
                m_templateStorage[id].returnSizes.end(), 0, [](const auto acc, const auto val) {
                    return acc + val.slots();
                });
            TemplateStorage::ConstT stack(totalReturnSize, 0ul);
            binTemplate.executeInplace(stack.data(), {0});
            setConstant(id, std::move(stack));
            INFO("template " << id << " is constant");
        }

        // do not store the template as it is not used again ...
        return;
    }

#ifdef FORCE_STACK_TEMPLATE
    ASSURE(static_cast<uint32_t>(kind), ==, 1);
#else
    ASSURE(static_cast<uint32_t>(kind), ==, 2);
#endif
    m_templateStorage[id].storage.emplace<TemplateStorage::TemplateT>(std::move(binTemplate));
}

void TemplateLibrary::setConstant(
    IdType id,
    std::vector<uint64_t>&& constant)
{
    ASSURE(id, <, m_templateStorage.size());
    m_templateStorage[id].storage.emplace<TemplateStorage::ConstT>(std::move(constant));
}

TemplateLibrary::Map::const_iterator TemplateLibrary::find(const KeyType& key) const
{
    MEASURE(ScopeTimer::templateLookup);
    return m_signatureToId.find(key);
}

TemplateLibrary::Map::const_iterator TemplateLibrary::find(const mlir::Operation& op) const
{
    MEASURE(ScopeTimer::templateLookup);
    return m_signatureToId.find_as(op);
}

TemplateLibrary::Map::const_iterator TemplateLibrary::end() const
{
    return m_signatureToId.end();
}

bool TemplateLibrary::contains(const KeyType& key) const
{
    return find(key) != end();
}

bool TemplateLibrary::contains(const mlir::Operation& op) const
{
    return find(op) != end();
}

bool TemplateLibrary::isConstant(IdType id) const
{
    ASSURE(id, <, m_templateStorage.size());
    return std::holds_alternative<TemplateStorage::ConstT>(m_templateStorage[id].storage);
}

std::tuple<
    const ResultSizes&,
    const StackMaps&,
    const std::vector<uint64_t>&>
TemplateLibrary::getConstant(IdType id) const
{
    ASSURE(id, <, m_templateStorage.size());
    const auto& elem = m_templateStorage[id];
    ASSURE_TRUE((std::holds_alternative<TemplateStorage::ConstT>(elem.storage)));
    return {elem.returnSizes, elem.stackMaps, std::get<TemplateStorage::ConstT>(elem.storage)};
}

std::tuple<
    const ResultSizes&,
    const StackMaps&,
    const Template&>
TemplateLibrary::getTemplate(IdType id) const
{
    ASSURE(id, <, m_templateStorage.size());
    const auto& elem = m_templateStorage[id];
    ASSURE_TRUE(std::holds_alternative<TemplateStorage::TemplateT>(elem.storage), << " missing template " << id);
    ASSURE_TRUE((std::get<TemplateStorage::TemplateT>)(elem.storage).isValid());
    return {elem.returnSizes, elem.stackMaps, std::get<TemplateStorage::TemplateT>(elem.storage)};
}

mlir::MLIRContext& TemplateLibrary::getContext()
{
    return m_context;
}

const std::string TemplateLibrary::getName() const
{
    return m_name;
}

TemplateLibrary::~TemplateLibrary()
{
    INFO(*this);
    dumpMetadata();

    for (auto* mem : m_aliveOps) {
        mem->destroy();
    }
}

void TemplateLibrary::dumpMetadata() const
{
    m_context.allowUnregisteredDialects();
    mlir::OpBuilder builder(&m_context);
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(moduleOp->getBody());

    for (const auto& sign_it : m_signatureToId) {
        const auto& signature = sign_it.first;
        const auto id = sign_it.second;
        const auto it = std::find_if(
            m_context.getRegisteredOperations().begin(),
            m_context.getRegisteredOperations().end(),
            [&](auto opname) {
                return opname.getTypeID() == signature.m_opId;
            });
        ASSURE(id, <, m_templateStorage.size());
        ASSURE(it, !=, m_context.getRegisteredOperations().end());
        std::vector<mlir::NamedAttribute> attributes;
        attributes.emplace_back(
            builder.getStringAttr("__signature_name__"),
            builder.getStringAttr(it->getStringRef()));
        attributes.emplace_back(
            builder.getStringAttr("__region_count__"),
            builder.getI8IntegerAttr(signature.m_numRegions));
        {
            std::vector<mlir::Attribute> typeSizeAttributes;
            for (const auto typeSize : m_templateStorage[id].returnSizes) {
                typeSizeAttributes.push_back(typeSize.ToAttr(builder));
            }
            attributes.emplace_back(
                builder.getStringAttr("__result_sizes__"),
                builder.getArrayAttr(typeSizeAttributes));
        }
        if (std::holds_alternative<Template>(m_templateStorage[id].storage)) {
            attributes.emplace_back(
                builder.getStringAttr("__clobber_mask__"),
                builder.getI16IntegerAttr(
                    std::get<Template>(m_templateStorage[id].storage).clobberMask.to_ulong()));
        }
        std::vector<mlir::Attribute> stackmapAttrs;
        for (const auto& map : m_templateStorage[id].stackMaps) {
            std::vector<mlir::Attribute> typeSizeAttributes;
            for (const auto typeSize : map) {
                typeSizeAttributes.push_back(typeSize.ToAttr(builder));
            }
            stackmapAttrs.push_back(builder.getArrayAttr(typeSizeAttributes));
        }
        attributes.emplace_back(
            builder.getStringAttr("__stackmap__"),
            builder.getArrayAttr(stackmapAttrs));
        attributes.emplace_back(
            builder.getStringAttr("__constexpr__"),
            builder.getBoolAttr(signature.isConstexpr()));
        attributes.emplace_back(
            builder.getStringAttr("__needsStack__"),
            builder.getBoolAttr(signature.needStack()));
        unsigned i = 0;
        for (const auto& op : signature.m_operands) {
            attributes.emplace_back(builder.getStringAttr("__opT" + std::to_string(i)), mlir::TypeAttr::get(op));
            i++;
        }
        mlir::Operation* tmp = mlir::Operation::create(
            builder.getUnknownLoc(),
            *it,
            {},
            {},
            {},
            signature.m_properties);
        auto propattr = tmp->getPropertiesAsAttribute();
        if (propattr)
            attributes.emplace_back(builder.getStringAttr("__properties__"), propattr);
        builder.insert(
            mlir::Operation::create(
                builder.getUnknownLoc(),
                mlir::OperationName("__signature__" + std::to_string(id), &m_context),
                signature.m_results,
                {},
                mlir::NamedAttrList{attributes},
                nullptr));
        tmp->destroy();
    }
    std::error_code errorCode;
    llvm::raw_fd_ostream file(m_name + "/.meta", errorCode);
    moduleOp->print(file);
    file.close();
}

void TemplateLibrary::readMetadata()
{
    const bool allowUnregistered = getContext().allowsUnregisteredDialects();
    getContext().allowUnregisteredDialects(true);

    llvm::SourceMgr sourceMgr;
    auto mod = mlir::parseSourceFile<mlir::ModuleOp>(m_name + "/.meta", sourceMgr, mlir::ParserConfig{&getContext()});
    const auto size = std::accumulate(mod->getOps().begin(), mod->getOps().end(), 1u, [](const auto& acc, auto&) {
        return acc + 1;
    });
    m_templateStorage.resize(size + m_templateStorage.size());
    std::vector<ClobberMask> clobbers;
    clobbers.resize(size + m_templateStorage.size());
    std::fill(clobbers.begin(), clobbers.end(), ClobberMask(CLOBBER_ALL));

    for (auto& op : mod->getOps()) {
        const auto name = op.getName().getStringRef();
        ASSURE_TRUE(name.starts_with("__signature__"));
        const auto id = static_cast<IdType>(std::stoi(name.substr(name.find_last_of("__") + 1).str()));
        auto resultAttr = op.getAttrOfType<mlir::ArrayAttr>("__result_sizes__");
        ResultSizes resultSizes;
        for (auto attr : resultAttr.getValue()) {
            resultSizes.push_back(TypeSize::FromAttr(attr));
        }

        auto stackmapAttr = op.getAttrOfType<mlir::ArrayAttr>("__stackmap__");
        StackMaps stackmaps;
        stackmaps.resize(stackmapAttr.size());
        std::transform(
            stackmapAttr.getAsRange<mlir::ArrayAttr>().begin(),
            stackmapAttr.getAsRange<mlir::ArrayAttr>().end(),
            stackmaps.begin(),
            [](const auto& array) {
                StackMap stackmap;
                for (auto subAttr : array.getValue()) {
                    stackmap.push_back(TypeSize::FromAttr(subAttr));
                }
                return stackmap;
            });

        const uint8_t regionCount = static_cast<uint8_t>(op.getAttrOfType<mlir::IntegerAttr>("__region_count__").getInt());
        const auto opName = op.getAttrOfType<mlir::StringAttr>("__signature_name__").strref();
        auto resultTypes = std::vector(op.getResultTypes().begin(), op.getResultTypes().end());
        std::vector<mlir::Type> operandTypes;
        for (unsigned i = 0; op.hasAttr("__opT" + std::to_string(i)); i++) {
            operandTypes.push_back(op.getAttrOfType<mlir::TypeAttr>("__opT" + std::to_string(i)).getValue());
        }
        const auto it = std::find_if(m_context.getRegisteredOperations().begin(), m_context.getRegisteredOperations().end(), [&](auto opname) {
            return opname.getStringRef() == opName;
        });
        ASSURE(it, !=, m_context.getRegisteredOperations().end());
        const bool constant = op.getAttrOfType<mlir::BoolAttr>("__constexpr__").getValue();
        const bool needsStack = op.getAttrOfType<mlir::BoolAttr>("__needsStack__").getValue();

        // we create the operation, however we can not insert it into a module as we might fail to verify some constraints
        // therefore we keep track of the operations our selves
        auto* tmp = mlir::Operation::create(
            mlir::UnknownLoc::get(&m_context),
            *it,
            {},
            {},
            {},
            nullptr);
        m_aliveOps.push_back(tmp);
        if (op.hasAttr("__properties__")) {
            bool ok = tmp->setPropertiesFromAttribute(op.getAttr("__properties__"), nullptr).succeeded();
            REL_ASSURE(ok, ==, true);
        }
        auto signature = Signature{
            it->getTypeID(),
            tmp->hashProperties(),
            tmp->getPropertiesStorage(),
            std::move(operandTypes),
            std::move(resultTypes),
            regionCount,
            constant,
            needsStack};
        std::construct_at(&m_templateStorage[id], std::move(resultSizes), std::move(stackmaps), mlir::ModuleOp{}, signature.isConstexpr());
        m_signatureToId[std::move(signature)] = id;
        if (op.getAttrOfType<mlir::IntegerAttr>("__clobber_mask__"))
            clobbers[id] = op.getAttrOfType<mlir::IntegerAttr>("__clobber_mask__").getValue().getZExtValue();
    }

    getContext().allowUnregisteredDialects(allowUnregistered);

    BinaryParser parser(*this);
    std::filesystem::path path(getName());
    for (const auto& file : std::filesystem::directory_iterator{path}) {
        const auto& p = file.path();
        if (p.extension() == ".o") {
            ASSURE_TRUE(isTemplateFunction(p.filename().string()));
            const uint32_t id = std::stoi(p.filename().string().substr(7));
            if (id >= CustomPatternGeneration::templateCount() && id < m_templateStorage.size())
                parser(p, clobbers[id]);
        }
    }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TemplateLibrary& lib)
{
    os << "\n\n============= Library =============\n";
    const auto sizeInBytes = [&os](const auto from, const auto to) {
        size_t sizeBin = 0;
        size_t sizeMeta = 0;
        for (auto i = from; i != to; i++) {
            if (std::holds_alternative<TemplateLibrary::TemplateStorage::TemplateT>(i->storage)) {
                sizeBin += std::get<TemplateLibrary::TemplateStorage::TemplateT>(i->storage).size();
            }
            sizeMeta += sizeof(TemplateLibrary::TemplateStorage)
                + i->returnSizes.size() * sizeof(int16_t)
                + i->stackMaps.size() * sizeof(std::vector<int32_t>)
                + std::accumulate(i->stackMaps.begin(), i->stackMaps.end(), 0, [](const auto acc, const auto val) {
                      return acc + val.size() * sizeof(int32_t);
                  });
        }
        os << "   - Binary   Size: " << sizeBin << " Bytes\n";
        os << "   - Metadata Size: " << sizeMeta << " Bytes\n";
    };
    const auto internal = CustomPatternGeneration::templateCount();
    os << "Internal: " << internal << " [0 - " << (internal - 1) << "]\n";
    sizeInBytes(lib.m_templateStorage.cbegin(), lib.m_templateStorage.cbegin() + internal);
    os << "Templates: " << (lib.m_templateStorage.size() - internal) << "\n";
    sizeInBytes(lib.m_templateStorage.cbegin() + internal, lib.m_templateStorage.cend());
    uint32_t i = 0, j = 0, k = 0;
    for (const auto& [sig, _] : lib.m_signatureToId) {
        i += (sig.id() == mlir::TypeID::get<mlir::LLVM::LLVMFuncOp>()) ? 1 : 0;
        j += (sig.id() == mlir::TypeID::get<mlir::LLVM::CallOp>()) ? 1 : 0;
        k += (sig.id() == mlir::TypeID::get<mlir::LLVM::GEPOp>()) ? 1 : 0;
    }
    os << "Functions: " << i << "\n";
    os << "Calls    : " << j << "\n";
    os << "GEPs     : " << k << "\n";
    return os;
}
