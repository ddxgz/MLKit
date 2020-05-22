import Logging
import TensorFlow

var logger = Logger(label: "dev.ml.swift.LinearModel")

protocol GDEstimator: LinearEstimator {
    var learningRate: Float { get set }
    var epochs: Int { get set }
    var model: Dense<Float>? { get set }
}

extension GDEstimator {
    var intercept_: Tensor<Float> {
        guard self.model != nil else { return Tensor<Float>(0) }
        return self.model!.bias
    }

    var coef_: Tensor<Float> {
        guard self.model != nil else { return Tensor<Float>(0) }
        return self.model!.weight
    }

    func predict(data x: Tensor<Float>) -> Tensor<Float> {
        guard self.model != nil else {
            logger.error("Model not trainied!")
            return Tensor<Float>(0)
        }

        Context.local.learningPhase = .inference
        return self.model!(x)
    }
}

// TODO: add config for activation, optimizer, loss
struct GDRegression: GDEstimator {
    var fitIntercept: Bool = true
    var scoring: String = "r2"
    var weights: Tensor<Float>

    var learningRate: Float
    var epochs: Int
    var model: Dense<Float>?

    init(learningRate: Float, epochs: Int, verbose: Bool = false) {
        self.learningRate = learningRate
        self.epochs = epochs

        self.weights = Tensor<Float>(0)
        // self.scoring = "r2"

        if verbose {
            logger.logLevel = .debug
        }
    }

    // @differentiable
    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>) {
        var model = Dense<Float>(inputSize: x.shape[1], outputSize: 1,
                                 activation: relu)

        let optim = SGD(for: model, learningRate: self.learningRate)
        // let optim = RMSProp(for: self.model, learningRate: self.learningRate)
        Context.local.learningPhase = .training

        for i in 1 ... self.epochs {
            let modelGrad = gradient(at: model) { model -> Tensor<Float> in
                let yhat = model(x)
                // let loss = l1Loss(predicted: yhat, expected: y)
                let loss = meanSquaredError(predicted: yhat, expected: y)
                // print("loss: \(loss)")
                logger.debug("Epoch: \(i), loss: \(loss)")
                return loss
            }
            optim.update(&model, along: modelGrad)
        }
        self.weights = model.weight.concatenated(
            with: model.bias.reshaped(to: [1, 1]),
            alongAxis: 0
        )
        self.model = model
    }
}
