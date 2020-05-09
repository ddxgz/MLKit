import TensorFlow


struct OLSRegression {
    var fitIntercept: Bool
    var weights: Tensor<Float>

    var intercept_: Tensor<Float> {
        get {
            if self.fitIntercept { 
                return self.weights[-1, 0...self.weights.shape[1]]
            } else { 
                return Tensor(0)
            }
        }
    }

    var coef_: Tensor<Float> {
        get {
            if self.fitIntercept { 
                return self.weights[
                    0..<self.weights.shape[0]-1, 0...self.weights.shape[1]]
            } else { 
                return self.weights
            }
        }
    }

    init(fitIntercept: Bool=true) {
        self.weights = Tensor<Float>(0)
        self.fitIntercept = fitIntercept
    }

    // \beta = (X^T dot X)^-1 dot X^T dot y
    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>) {
        var x = x

        if self.fitIntercept {
            x = x.concatenated(with: Tensor<Float>(ones: [x.shape[0], 1]), 
                                alongAxis: -1)
        }

        self.weights = matmul(
                            matmul(
                                _Raw.matrixInverse(
                                    matmul(x.transposed(), x)),
                                x.transposed()), 
                            y)
    }

    func predict(data x: Tensor<Float>) -> Tensor<Float> {
        let x = self.preprocessX(x)
        return matmul(x, self.weights)
    }

    // r^2
    func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float {
        let predicted = self.predict(data: x)
        let SS_res = pow(y - predicted, 2).sum()
        print(SS_res)
        let SS_tot = pow(y - y.mean(), 2).sum()
        print(SS_tot)
        print(SS_res/SS_tot)
        let score = 1 - (SS_res / SS_tot)
        return Float(score.scalarized())
    }


    func preprocessX(_ x: Tensor<Float>) -> Tensor<Float> {
        var x = x

        if self.fitIntercept {
            x = x.concatenated(with: Tensor<Float>(ones: [x.shape[0], 1]), 
                                alongAxis: -1)
        }
        return x
    }
}