// import LASwift
import TensorFlow

public protocol PCATransformer: FitTransformer {
    var components: Matrix? { get }
}

public enum SvdSolver: String {
    case full
}

public struct PCA: PCATransformer {
    public let nComponents: Int
    let svdSolver: SvdSolver

    var mean: Matrix?
    public var nSamples: Int?
    public var nFeatures: Int?
    public var noiseVariance: Float?
    public var components: Matrix?
    public var explainedVariance: Matrix?
    public var explainedVarianceRatio: Matrix?
    public var singularValues: Matrix?

    public init(nComponents: Int, svdSolver: String) throws {
        guard nComponents > 0 else {
            throw EstimatorError.notSupportedParameter("nComponents should be > 0")
        }
        self.nComponents = nComponents

        let svdSolverOpt = SvdSolver(rawValue: svdSolver)
        guard svdSolverOpt != nil else {
            throw EstimatorError.notSupportedParameter("not supported svd solver!")
        }
        self.svdSolver = svdSolverOpt!
    }

    // public func fit(data x: Matrix, labels y: Matrix) -> PCATransformer {
    public mutating func fit(_ x: Matrix) -> FitTransformer {
        self._fit(x)
        return self
    }

    mutating func _fit(_ x: Matrix) -> (u: Tensor<Float>, s: Tensor<Float>, v: Tensor<Float>) {
        // TODO: check if X is sparse matrix
        // TODO: validate data

        switch self.svdSolver {
        case .full:
            return self.fitFull(x)
        }
        // TODO: other solvers, e.g., randomized
    }

    mutating func fitFull(_ x: Matrix) -> (u: Tensor<Float>, s: Tensor<Float>, v: Tensor<Float>) {
        let nSamples = x.shape[0]
        let nFeatures = x.shape[1]

        // center data
        self.mean = x.mean(alongAxes: Tensor<Int32>(0))
        let centeredX = x - self.mean!
        // print(x)
        // print(centeredx)

        // let (u, s, v) = _Raw.svd(center, fullMatrices: false)
        // print(Matrix(v.transposed())[0..., 0 ..< 3])
        let (S, U, V) = _Raw.svd(centeredX)

        // print(V)
        // TODO: flip eigenvector's sign to enforce deterministic output
        let (UFlipped, VFlipped) = svdFlip(u: U, v: V.transposed(), uBasedDecision: true)
        print(VFlipped)

        let explainedVariance = pow(S, 2) / Float(nSamples - 1)
        let totalVar = explainedVariance.sum()
        let explainedVarianceRatio = explainedVariance / totalVar
        let singularValues = S

        if self.nComponents < min(nFeatures, nSamples) {
            self.noiseVariance = Float(explainedVariance[self.nComponents...].mean())
        } else {
            self.noiseVariance = 0.0
        }

        // self.components = V.transposed()[0 ..< self.nComponents]
        self.components = VFlipped[0 ..< self.nComponents]
        // self.components = Matrix(V[0 ..< self.nComponents])
        self.nSamples = nSamples
        self.nFeatures = nFeatures
        self.explainedVariance = explainedVariance
        self.explainedVarianceRatio = explainedVarianceRatio
        self.singularValues = singularValues

        return (UFlipped, S, VFlipped)
    }

    // TODO:
    public func transform(_ x: Matrix) throws -> Matrix {
        // TODO: check if self is fitted
        guard self.components != nil else {
            throw EstimatorError.notFittedError()
        }

        // TODO: check input

        var X: Matrix
        if self.mean != nil {
            X = x - self.mean!
        } else {
            X = x
        }
        let xTransformed = matmul(X, self.components!.transposed())

        // TODO: if self.whiten
        return xTransformed
    }

    // TODO:
    public mutating func fitTranform(_ x: Matrix) -> Matrix {
        let decomposed = self._fit(x)
        let U = decomposed.u[0..., 0 ..< self.nComponents]
        // TODO: if self.whiten

        return U * decomposed.s[0 ..< self.nComponents]
    }
}

public func svdFlip(u: Matrix, v: Matrix, uBasedDecision: Bool = true) -> (Matrix, Matrix) {
    if uBasedDecision {
        let maxAbsCols = _Raw.abs(u).argmax(squeezingAxis: 0)
        let maxpos = maxAbsCols.expandingShape(at: 1)
        let withmaxcols = u.transposed().batchGathering(atIndices: maxpos,
                                                        alongAxis: 1).transposed()
        let signs = _Raw.sign(withmaxcols)
        let uFlipped = u * signs
        let vFlipped = v * signs.transposed()

        return (uFlipped, vFlipped)
    } else {
        let maxAbsCols = _Raw.abs(v.transposed()).argmax(squeezingAxis: 0)

        let maxpos = maxAbsCols.expandingShape(at: 1)
        let withmaxcols = v.batchGathering(atIndices: maxpos,
                                           alongAxis: 1).transposed()
        let signs = _Raw.sign(withmaxcols)
        let uFlipped = u * signs
        let vFlipped = v * signs.transposed()
        return (uFlipped, vFlipped)
    }
}
