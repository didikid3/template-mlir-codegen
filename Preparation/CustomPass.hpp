#pragma once

#include "common.hpp"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

struct ToLLVMLoweringPass
    : public mlir::PassWrapper<ToLLVMLoweringPass, mlir::OperationPass<>>
{
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;

    static std::unique_ptr<mlir::Pass> create();
};

/**
 * @brief Rewrite input unrealized-conversion-casts
 *
 * @details
 * Rewrites unrealized-conversion-casts for input operands to the respective concrete implementation
 * associated with the template kind
 */
class LoadInputFromStackRewriteOp : public mlir::RewritePattern
{
    const TEMPLATE_KIND m_kind;

public:
    LoadInputFromStackRewriteOp(mlir::MLIRContext* context, TEMPLATE_KIND kind);

    mlir::LogicalResult match(mlir::Operation* op) const override;
    void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override;
};

/**
 * @brief Rewrite output/store unrealized-conversion-casts
 *
 * @details
 * Rewrites unrealized-conversion-casts for results to the respective concrete implementation
 * associated with the template kind
 */
class StoreRegionArgsToStackRewriteOp : public mlir::RewritePattern
{
    const TEMPLATE_KIND m_kind;

public:
    StoreRegionArgsToStackRewriteOp(mlir::MLIRContext* context, TEMPLATE_KIND kind);

    mlir::LogicalResult match(mlir::Operation* op) const override;
    void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override;
};

/**
 * @brief Add tail-calls at the end of a function if requested with `kTailCallIndicator`
 */
class TailCallRewriteOp : public mlir::RewritePattern
{
    const TEMPLATE_KIND m_kind;

public:
    TailCallRewriteOp(
        mlir::MLIRContext* context,
        TEMPLATE_KIND kind);

    mlir::LogicalResult match(mlir::Operation* op) const override;
    void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override;
};
