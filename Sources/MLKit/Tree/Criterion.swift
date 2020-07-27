import TensorFlow

/// A criterion implementation has a function nodeImpurity that takes input of
/// groups of labels (either classes or
/// numerical values) in the form of array of 2 arraies [left, right]
protocol Criterion {
    func nodeImpurity(_: [[Float]]) -> Float
}

struct Gini: Criterion {
    func nodeImpurity(_ groups: [[Float]]) -> Float {
        let classes = Set(groups.joined())
        // print("classes: \(classes)")
        var N: Float = 0.0
        for group in groups {
            N += Float(group.count)
            // N += Double(group.count)
        }
        // print(N)
        var gini: Float = 0.0
        for group in groups {
            let sizeG = Float(group.count)
            if sizeG == 0 { continue }
            /// sum((n_k/N)^2) for class k
            var sumP: Float = 0.0
            for cls in classes {
                // let p = group.count(where: $0 == cls) / sizeG
                let p = Float(group.filter { $0 == cls }.count) / sizeG
                // print(p)
                sumP += p * p
            }
            gini += (1.0 - sumP) * (sizeG / N)
        }
        return Float(gini)
    }
}

struct MSE: Criterion {
    func nodeImpurity(_ groups: [[Float]]) -> Float {
        var N: Float = 0.0
        for group in groups {
            N += Float(group.count)
        }

        var err: Float = 0

        for group in groups {
            let sizeG = Float(group.count)
            if sizeG == 0 { continue }
            let mean = group.reduce(0, +) / sizeG
            // // print("group: \(group)")
            // // print("mean: \(mean)")
            // // use a tensor fill with mean
            // let y = Tensor<Float>(repeating: mean, shape: [Int(sizeG), 1])
            // let yhat = Tensor<Float>(group)
            // err += meanSquaredErrorTF(y, yhat)
            let diff = group.map { pow($0 - mean, 2) }
            err += diff.reduce(0, +) / sizeG
            // print("err: \(err)")
        }
        return err
    }
}

struct MSEv1: Criterion {
    func nodeImpurity(_ groups: [[Float]]) -> Float {
        var N: Float = 0.0
        for group in groups {
            N += Float(group.count)
        }

        var err: Float = 0

        for group in groups {
            let sizeG = Float(group.count)
            if sizeG == 0 { continue }
            let mean = group.reduce(0, +) / sizeG
            // print("group: \(group)")
            // print("mean: \(mean)")
            // use a tensor fill with mean
            let y = Tensor<Float>(repeating: mean, shape: [Int(sizeG), 1])
            let yhat = Tensor<Float>(group)
            err += meanSquaredErrorTF(y, yhat)
            // print("err: \(err)")
        }
        return err
    }
}

/// Use enum to represent set of supported criterion
func getCriterion(_ type_: String) -> Criterion? {
    switch type_ {
    case "gini":
        return Gini()
    case "mse":
        return MSE()
    case "msev1":
        return MSEv1()
    default:
        print("Criterion: \(type_) not supported!")
        return nil
    }
}
