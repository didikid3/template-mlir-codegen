#pragma once

#include "Preparation/BinaryParser.hpp"
#include "TemplateLibrary.hpp"

namespace mlir {
class ModuleOp;
class Operation;
class MLIRContext;
}

class TemplateBuilder
{
public:
    TemplateBuilder(
        TemplateLibrary& library,
        const Abstraction::IRLoweringFunction& lowering);

    void addOp(mlir::Operation& op);
    void addOpWithSignature(
        mlir::Operation& op,
        Abstraction abstraction,
        Signature&& signature);
    void addOpsFromModule(mlir::ModuleOp& mod);
    void addOpsFromFile(const std::string& filename);

    mlir::MLIRContext& getContext();

private:
    friend void parseAndRegisterTemplate(std::string, TemplateBuilder&, uint32_t);
    template <typename>
    friend struct CustomPattern;

    TemplateLibrary::IdType reserveInternal();
    void compileInternal(
        TemplateLibrary::IdType id,
        mlir::ModuleOp& mod);

    TemplateLibrary& m_library;
    BinaryParser m_parser;
    const Abstraction::IRLoweringFunction m_lowering;
};
