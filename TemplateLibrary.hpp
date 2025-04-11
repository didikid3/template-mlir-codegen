#pragma once

#include "Preparation/Abstraction.hpp"
#include "Preparation/SectionProvider.hpp"
#include "Templates/Signature.hpp"
#include "Templates/Template.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/DenseMap.h"

#include <tuple>
#include <variant>
#include <vector>

/**
 * @brief Collection of prepared templates
 *
 * The template library represents the collection of prepared templates.
 * New templates should be added via a `TemplateBuilder`
 *
 * @details
 * Default workflow:
 *  #1 create a template library (an empty one with `Empty` or parse an existing folder with contained templates with `FromFolder`)
 *  #2 use a `TemplateBuilder` add new templates during preparation
 *  #3 lookup templates (`find`) to get the respective `IdType` (or not found)
 *     !! use the cheaper versions of `find`, that directly work on an operation instead of first constructing a signature !!
 *  #4 retrieve the template or meta information (`getTemplate`, `isConstant` etc.)
 */
class TemplateLibrary
    : public CodeSectionProvider
    , public DataSectionProvider
{

public:
    using KeyType = Signature;
    using IdType = uint32_t;
    using Map = llvm::DenseMap<KeyType, IdType>;

    TemplateLibrary(TemplateLibrary&& rhs) = default;
    TemplateLibrary(const TemplateLibrary& rhs) = delete;
    TemplateLibrary(TemplateLibrary& rhs) = delete;
    ~TemplateLibrary();

    static TemplateLibrary Empty(
        const std::string& name,
        mlir::MLIRContext& context);
    static TemplateLibrary FromFolder(
        const std::string& folder,
        mlir::MLIRContext& context);

    void setConstant(
        IdType id,
        std::vector<uint64_t>&& constant);
    Map::const_iterator find(const KeyType& key) const;
    Map::const_iterator find(const mlir::Operation& op) const;
    Map::const_iterator end() const;
    bool contains(const KeyType& key) const;
    bool contains(const mlir::Operation& key) const;
    bool isConstant(IdType id) const;
    std::tuple<
        const ResultSizes&,
        const StackMaps&,
        const std::vector<uint64_t>&>
    getConstant(IdType id) const;

    std::tuple<
        const ResultSizes&,
        const StackMaps&,
        const Template&>
    getTemplate(IdType id) const;

    mlir::MLIRContext& getContext();
    const std::string getName() const;

private:
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TemplateLibrary& s);
    friend class TemplateBuilder;
    friend class BinaryParser;

    TemplateLibrary(
        const std::string& name,
        mlir::MLIRContext& context);

    void readMetadata();
    void dumpMetadata() const;

    IdType insert(
        KeyType&& signature,
        ResultSizes&& returnSizes,
        StackMaps&& stackMaps,
        const mlir::ModuleOp& function /*the stack args function*/);

    void addBinaryTemplate(
        IdType id,
        TEMPLATE_KIND kind,
        Template&& binTemplate);

    struct TemplateStorage
    {
        using ConstT = std::vector<uint64_t>;
        using ModT = mlir::OwningOpRef<mlir::ModuleOp>;
        using TemplateT = Template;

        ResultSizes returnSizes;
        StackMaps stackMaps;
        std::variant<
            ModT,
            ConstT,
            TemplateT>
            storage;
        bool isConst;

        TemplateStorage(
            ResultSizes&& returnSizes_,
            StackMaps&& stackMaps_,
            const mlir::ModuleOp& module,
            bool isConst)
            : returnSizes(std::move(returnSizes_))
            , stackMaps(std::move(stackMaps_))
            , storage(module)
            , isConst(isConst)
        {
        }

        TemplateStorage() = default;
    };

    const std::string m_name;
    mlir::MLIRContext& m_context;
    std::vector<TemplateStorage> m_templateStorage;
    Map m_signatureToId;
    std::vector<mlir::Operation*> m_aliveOps;
};
