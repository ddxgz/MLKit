import TensorFlow

#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

// let np = Python.import("numpy")

// typealias IntValue = Int32

public protocol TreeEstimator: Estimator, Predictor {
    var criterion: String { get set }
    var nFeatures: Int { get set }
    var nClasses: Int { get set }
    var maxDepth: Int { get set }
    var maxFeatures: Int { get set }
    var minSamplesSplit: Int { get set }
    var minSamplesLeaf: Int { get set }
    var featureImportances: Tensor<Float> { get }
    var tree: DTree? { get set }
    // var scoring: String { get }

    // mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>)

    // func predict(data x: Tensor<Float>) -> Tensor<Float>

    // func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float
}

typealias Groups = (left: [Int], right: [Int])
typealias LabelGroups = (left: [Float], right: [Float])

class Node: CustomStringConvertible {
    // var id: Int
    var isEmpty: Bool
    var leftChild: Int?
    var rightChild: Int?
    var depth: Int
    var feature: Int
    var splitValue: Float
    // var impurity: Double
    var score: Float
    // var nSamples: Int64
    var isLeaf: Bool = false
    var value: Float?
    var groups: Groups

    var nSamples: Int { groups.left.count + groups.right.count }

    init(depth: Int, feature: Int, splitValue: Float, score: Float,
         groups: Groups) {
        self.isEmpty = false
        // self.id = id
        self.depth = depth
        self.feature = feature
        self.splitValue = splitValue
        self.score = score
        self.groups = groups
    }

    init(isEmpty: Bool) {
        self.isEmpty = true
        // self.id = -1
        self.depth = -1
        self.feature = -1
        self.splitValue = -1
        self.score = -1
        self.groups = (left: [Int](), right: [Int]())
    }

    public var description: String {
        if isEmpty { return "[empty node]" }

        let space = String(repeating: " ", count: depth * 2)
        // var desc = "\(space)id: \(id), leaf: \(isLeaf), "
        var desc = "\(space)leaf: \(isLeaf), "

        // let possibleValue = value ?? 0.0
        if isLeaf { desc += "value: \(value!), n_samples: \(nSamples), " } else {
            desc += "left child: \(leftChild!), right child: \(rightChild!), "
            desc += "groups: [\(groups.left.count) \(groups.right.count)], "
        }

        return desc + """
        feature: \(feature), splitValue: \(splitValue), score: \(score), 
        """
    }
}

/// The root N of T is stored in TREE [1].
/// If a node occupies TREE [k] then its left child is stored in TREE [2 * k]
/// and its right child is stored into TREE [2 * k + 1].
public class DTree {
    var nodes: [Node]
    init() { nodes = [Node]() }

    func addNode(_ node: Node) {
        // nodes.append(node)
        // print("nodes count: \(nodes.count)")
        // print("nodes cap: \(nodes.capacity)")
        // print("adding node: \(node.id)")
        // if (node.id - 1) >= nodes.count {
        //     // nodes.reserveCapacity(node.id * 2 + 1)
        //     paddingEmptyNodes(node.id * 2 + 1)
        // }

        // print("nodes count: \(nodes.count)")
        // nodes.insert(node, at: node.id - 1)
        nodes.append(node)
    }

    func addNode(_ node: Node, parent: Node?, isLeft: Bool) {
        if parent != nil {
            let nodeId = nodes.count
            if isLeft {
                parent!.leftChild = nodeId
            } else {
                parent!.rightChild = nodeId
            }
        }
        nodes.append(node)
    }

    // func addLeftChild(parent: Node, child: Node) {
    //     child.id = parent.id * 2
    //     parent.leftChild = child.id
    //     nodes.insert(child, at: parent.id * 2 - 1)
    // }

    // func addRightChild(parent: Node, child: Node) {
    //     child.id = parent.id * 2 + 1
    //     parent.rightChild = child.id
    //     nodes.insert(child, at: parent.id * 2)
    // }

    func paddingEmptyNodes(_ size: Int) {
        // print("padding emptyNodes")
        let padSize = size - nodes.count
        let emptyNodes = Array(repeating: Node(isEmpty: true), count: padSize)
        // print(emptyNodes.count)
        nodes += emptyNodes
    }

    func predict(_ x: Matrix) -> [Float] {
        let n_samples = x.shape[0]
        // var out = Matrix(shape:[1, 1])
        var out = [Float](repeating: -1, count: n_samples)
        // var new: Matrix = Matrix(shape: [1, data.shape[1]], repeating: 0) // = NdArray()
        for i in 0 ..< n_samples {
            var node = nodes[0]
            // print("sample: \(i), \(x[i])")
            // while leftChild(node).isLeaf != true {
            while node.isLeaf != true {
                // print("in node: \(node)")
                // print("data value to compare: \(x[i, node.feature].scalar)")
                if x[i, node.feature].scalar! <= node.splitValue {
                    node = leftChild(node)
                } else {
                    node = rightChild(node)
                }
            }
            // out = out.concatenated(with: node.value, alongAxis: 0)
            // print(node)
            out[i] = node.value!
        }
        // print(out)
        // return out[1..., 0...]
        return out
    }

    func leftChild(_ parent: Node) -> Node {
        // return nodes[parent.id * 2 - 1]
        return nodes[parent.leftChild!]
    }

    func rightChild(_ parent: Node) -> Node {
        // return nodes[parent.id * 2]
        return nodes[parent.rightChild!]
    }
}

func notConstant(_ data: Matrix) -> Bool {
    let vset = Set(data.scalars)
    if vset.count == 1 {
        print(data)
        return false
    }
    return true
}

struct BestFirstTreeBuilder {
    // let criterion: CriterionFn
    let criterion: Criterion
    let isClassification: Bool
    var nFeatures: Int
    var classes: [Int]
    var nOutputs: Int
    var maxDepth: Int
    var minSamplesSplit: Int
    var minSamplesLeaf: Int
    var maxFeatures: Int
    // var criterionFn: CriterionFn

    func build(data dataIn: Matrix) -> DTree {
        var colsUse = [Int]()
        for col in 0 ..< dataIn.shape[1] {
            if notConstant(dataIn[0..., col]) { colsUse.append(col) }
        }
        let data = dataIn.select(cols: colsUse)

        let tree = DTree()
        let depth = 0
        let node = addSplitNode(tree: tree, data: data, depth: depth, isFirst: true, isLeft: nil, parent: nil)
        tree.addNode(node)

        splitNode(tree: tree, node: node, data: data, depth: depth + 1)

        return tree
    }

    // TODO: impurity improvement
    func splitNode(tree: DTree, node: Node, data: Matrix, depth: Int) {
        if isLeaf(node) {
            // print("rearched Leaf")
            markLeaf(node, data: data)
            return
        }
        let (left, right) = node.groups
        // let leftData = dataSample(idx: left, data: data)
        let leftData = data.select(rows: left)
        let leftNode = addSplitNode(tree: tree, data: leftData, depth: depth, isFirst: false, isLeft: true, parent: node)
        // tree.addLeftChild(parent: node, child: leftNode)
        // tree.addNode(leftNode)

        // let rightData = dataSample(idx: right, data: data)
        let rightData = data.select(rows: right)
        let rightNode = addSplitNode(tree: tree, data: rightData, depth: depth, isFirst: false, isLeft: false, parent: node)
        // tree.addRightChild(parent: node, child: rightNode)
        // tree.addNode(rightNode)

        splitNode(tree: tree, node: leftNode, data: leftData, depth: depth + 1)
        splitNode(tree: tree, node: rightNode, data: rightData, depth: depth + 1)
    }

    /// replaced by Matrix.select
    func dataSample(idx: [Int32], data: Matrix) -> Matrix {
        var new: Matrix = Matrix(repeating: 0, shape: [1, data.shape[1]]) // = NdArray()
        for i in idx {
            // let data = x.concatenated(with: y1d, alongAxis: 1)
            let row = data[Int(i)].reshaped(to: [1, data.shape[1]])
            // let y1d = y.reshaped(to: [y.shape[0], 1])
            // print("new: \(new.shape), row: \(row.shape)")
            new = new.concatenated(with: row, alongAxis: 0)
            // new.replacing(with: row, where: new[Int(i), 0...].Index)
            // print(row)
        }
        // print(new[1..., 0...])
        // return data
        return new[1..., 0...]
    }

    func markLeaf(_ node: Node, data: Matrix) {
        let (left, right) = node.groups
        let combined = left + right

        var values = [Float]()

        for idx in combined {
            let c = data[Int(idx), -1]
            values.append(c.scalar!)
        }
        // print("values: \(values)")
        if isClassification {
            // node.value = values.max
            let cnt = values.reduce(into: [:]) { counts, number in
                counts[number, default: 0] += 1
            }
            // print("cnt: \(cnt)")
            let (value, _) = cnt.max(by: { a, b in a.value < b.value })!
            // print("value: \(value)")
            node.value = value
        } else {
            let sum = values.reduce(0, +)
            node.value = sum / Float(values.count)
        }
        // print(node.value)
        // TODO: regression
        node.isLeaf = true
    }

    func isLeaf(_ node: Node) -> Bool {
        // guard node.groups != nil else { return true }

        // print(node)
        /// For both classification and regression. Either the node is pure or
        /// no variance in node, it should be a leaf.
        if node.score == 0 {
            return true
        }
        // if node.groups.left == nil || node.groups.right == nil {
        if node.groups.left.count == 0 || node.groups.right.count == 0 {
            print(node.groups)
            return true
        }
        // if (node.groups.left.count + node.groups.right.count) < 2 * minSamplesLeaf {
        //     return true
        // }
        if (node.groups.left.count + node.groups.right.count) <= minSamplesSplit {
            return true
        }
        if node.depth >= maxDepth {
            return true
        }

        return false
    }

    func addSplitNode(tree: DTree, data: Matrix, depth: Int, isFirst: Bool, isLeft: Bool?, parent: Node?) -> Node {
        var bstScore: Float?
        var bstCol: Int = 0
        var bstSplitValue: Float = -1
        var bstGroups: Groups?
        // var nodeId: Int = 0

        // print(data.shape)
        if data.shape[0] <= minSamplesLeaf {
            var left = [Int](), right = [Int]()
            for (idx, value) in data[0..., 0].scalars.enumerated() {
                let rowIdx = Int(idx)
                left.append(rowIdx)
            }
            let sampleSplit = (left, right)

            let labelGroups = getLabelGroups(sampleSplit: sampleSplit, data: data)
            // let score = criterion([labelGroups.left, labelGroups.right])
            let score = criterion.nodeImpurity([labelGroups.left, labelGroups.right])

            let node = Node(depth: depth, feature: -1, splitValue: bstSplitValue, score: score,
                            groups: sampleSplit)
            if let unwrapLeft = isLeft {
                tree.addNode(node, parent: parent, isLeft: unwrapLeft)
            }
            return node
        }

        for col in 0 ..< nFeatures {
            // print(col)
            let s = data[0..., col].scalars
            let argsortIdx = s.indices.sorted { s[$0] < s[$1] }

            for (i, rowIdx) in argsortIdx.enumerated() {
                if i < minSamplesLeaf {
                    continue
                }

                let valueTensor = data[Int(rowIdx), col]
                // print("before grt value")
                let value = valueTensor.scalar!
                // print("after grt value")

                // let sampleSplit = getSampleSplit(col: col, splitBy: value, data: data)
                let sampleLeft = [Int](argsortIdx[0 ..< i])
                let sampleRight = [Int](argsortIdx[i...])
                let sampleSplit: Groups = (sampleLeft, sampleRight)
                // let labelLeft = data.select(row: sampleLeft, col: -1)
                // let labelRight = data.select(row: sampleRight, col: -1)

                // print("sampleSplit: \(sampleSplit)")
                let labelGroups = getLabelGroups(sampleSplit: sampleSplit, data: data)

                // print("labelGroups: \(labelGroups)")
                // let score = criterion([labelGroups.left, labelGroups.right])
                let score = criterion.nodeImpurity([labelGroups.left, labelGroups.right])

                // let score = criterion([labelLeft, labelRight])

                // print("score: \(score)")
                // print("bstScore: \(bstScore)")
                if bstScore == nil || score < bstScore! {
                    bstScore = score
                    bstCol = col
                    bstSplitValue = value
                    bstGroups = sampleSplit
                    // bstGroups = (left: sampleLeft, right: sampleRight)
                }
                if i >= Int(argsortIdx.count) - minSamplesLeaf - 1 {
                    break
                }

                if bstScore == 0 { break }
            }
            if bstScore == 0 { break }
        }
        // if isFirst {
        //     nodeId = 1
        // } else if parent != nil, isLeft != nil {
        //     if isLeft! {
        //         nodeId = parent!.id * 2
        //     } else {
        //         nodeId = parent!.id * 2 + 1
        //     }
        // } else {
        //     print("Should provide isLeft and parent node if not first!")
        // }
        // print("bstGroups: \(bstGroups)")

        let node = Node(depth: depth, feature: bstCol, splitValue: bstSplitValue, score: bstScore!,
                        groups: bstGroups!)
        if let unwrapLeft = isLeft {
            tree.addNode(node, parent: parent, isLeft: unwrapLeft)
        }
        return node
    }

    // TODO: handle when the splitBy is the smallest or largest, ont of the
    /// group will be empty
    func getSampleSplit(col: Int, splitBy: Float, data: Matrix) -> Groups {
        var left = [Int](), right = [Int]()
        for (idx, value) in data[0..., col].scalars.enumerated() {
            let rowIdx = Int(idx)

            if value <= splitBy {
                left.append(rowIdx)
            } else {
                right.append(rowIdx)
            }
        }
        return (left, right)
    }

    func getLabelGroups(sampleSplit idxGroups: Groups, data: Matrix) -> LabelGroups {
        // var left = [Int](), right = [Int]()
        var left = [Float](), right = [Float]()
        for idx in idxGroups.left {
            let c = data[Int(idx), -1]
            // let v = Int(c.scalar!)
            let v = c.scalar!
            left.append(v)
        }
        for idx in idxGroups.right {
            let c = data[Int(idx), -1]
            // let v = Int(c.scalar!)
            let v = c.scalar!
            right.append(v)
        }

        return (left, right)
    }
}

extension TreeEstimator {
    public mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>) {
        //// check input data is 2d
        // print(x.shape)

        let nSamples = Int(x.shape[0])
        self.nFeatures = Int(x.shape[1])
        precondition(nSamples > 0, "n_samples: \(nSamples) <= 0")
        assert(nFeatures > 0, "n_features: \(nFeatures) <= 0")
        precondition(y.shape[0] == nSamples, """
        Number of labels: \(y.shape[0]) \
        dose not match number of n_samples: \(nSamples)!
        """)

        precondition(y.shape.count == 1 || y.shape[1] == 1, """
        Number of columns of labels: \(y.shape[1]) is not supported yet!
        """)
        // print(y.shape)

        print("nSamples \(nSamples) \(type(of: nSamples))")
        print("nFeatures \(nFeatures) \(type(of: nFeatures))")

        /// encode classes
        // var yEncoded = Tensor<Int32>(zeros: [y.shape[0], 1])
        // print(yEncoded.reshaped(to:[-1,1]))
        // let yFlat = y.reshaped(to: [-1])
        // let elements = _Raw.unique(y)
        var classes: Tensor<Float>, yEncoded: Tensor<Int32>
        (classes, yEncoded) = _Raw.unique(y.reshaped(to: [-1]))
        // let classes, yEncoded = elements.y, elements.idx
        // print("classes: \(classes)\nyEncoded: \(yEncoded)")

        let classesEnc = classes.scalars.map { Int($0) }
        print(type(of: classesEnc))

        nClasses = Int(classes.shape[0])
        let nOutputs = Int(yEncoded.shape[0])

        //// check parameters
        if maxDepth == -1 {
            // maxDepth = 9999
            maxDepth = Int.max
        }
        print("maxDepth \(maxDepth)")

        if maxFeatures == -1 || maxFeatures > nFeatures {
            maxFeatures = nFeatures
        }

        // guard let fnCriterion = Criteria[criterion] else {
        //     print("criterion not found")
        //     return
        // }
        guard let fnCriterion = getCriterion(criterion) else {
            print("criterion not found")
            return
        }

        var isClassification: Bool
        if criterion == "gini" {
            isClassification = true
        } else {
            isClassification = false
        }
        print("isClassification: \(isClassification)")

        /// wrap x and y to a 2d tensor, with y as -1 col
        let y1d = y.reshaped(to: [y.shape[0], 1])
        let data = x.concatenated(with: y1d, alongAxis: 1)
        // let data = _Raw.concat(concatDim: Tensor<Int32>(1), [x, y])
        // var data: [[Float]]

        // for row in x.array{

        // }
        // print(type(of:data.array))

        let builder = BestFirstTreeBuilder(criterion: fnCriterion,
                                           isClassification: isClassification,
                                           nFeatures: nFeatures,
                                           classes: classesEnc, nOutputs: nOutputs,
                                           maxDepth: maxDepth,
                                           minSamplesSplit: self.minSamplesSplit,
                                           minSamplesLeaf: self.minSamplesLeaf,
                                           maxFeatures: maxFeatures)

        tree = builder.build(data: data)
        // print(tree.score)
    }

    public func predict(data x: Tensor<Float>) -> Tensor<Float> {
        // print(x)
        let proba = tree!.predict(x)
        print(proba)
        // let result = Tensor<Float>(proba)
        let result = Tensor<Float>(proba).reshaped(to: [proba.count, 1])
        // print(type(of:result))
        // print(result.shape)
        return result
    }

    public func printTree() {
        guard self.tree != nil else {
            print("Tree not built!")
            return
        }
        let tree = self.tree!
        // printNode(node: node, depth: 0)
        // print(tree.nodes)
        for (i, node) in tree.nodes.enumerated() {
            if node.isEmpty { continue }
            print(i, node)
        }
    }
}

public struct DecisionTreeClassifier: TreeEstimator {
    // var criterion, splitter: String
    public var criterion: String
    public var nFeatures: Int = 0
    public var nClasses: Int = 0
    public var maxDepth: Int
    public var maxFeatures: Int
    public var minSamplesSplit: Int
    public var minSamplesLeaf: Int
    /// Impurity threshold used for split early stop
    // TODO: not yet really supported!
    var minImpurityDecrease: Float
    public var tree: DTree?
    public var featureImportances: Tensor<Float> { return Tensor(0) }
    public var scoring: String = "accuracy"

    //  splitter: String = "best",
    public init(criterion: String = "gini",
                maxDepth: Int = -1,
                maxFeatures: Int = -1, minSamplesSplit: Int = 2, minSamplesLeaf: Int = 1,
                minImpurityDecrease: Float = 0,
                scoring: String = "accuracy") {
        // (self.criterion, self.splitter) = (criterion, splitter)
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        if minImpurityDecrease < 0 {
            print("minImpurityDecrease must be >= 0, it is now set to 0")
            self.minImpurityDecrease = 0
        } else {
            self.minImpurityDecrease = minImpurityDecrease
        }
        self.scoring = scoring
    }
}

public struct DecisionTreeRegressor: TreeEstimator {
    // var criterion, splitter: String
    public var criterion: String
    public var nFeatures: Int = 0
    public var nClasses: Int = 0
    public var maxDepth: Int
    public var maxFeatures: Int
    public var minSamplesSplit: Int
    public var minSamplesLeaf: Int
    /// Impurity threshold used for split early stop
    // TODO: not yet really supported!
    var minImpurityDecrease: Float
    public var tree: DTree?
    public var featureImportances: Tensor<Float> { return Tensor(0) }
    public var scoring: String = "accuracy"

    //  splitter: String = "best",
    public init(criterion: String = "mse",
                maxDepth: Int = -1,
                maxFeatures: Int = -1, minSamplesSplit: Int = 2, minSamplesLeaf: Int = 1,
                minImpurityDecrease: Float = 0,
                scoring: String = "r2") {
        // (self.criterion, self.splitter) = (criterion, splitter)
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        if minImpurityDecrease < 0 {
            print("minImpurityDecrease must be >= 0, it is now set to 0")
            self.minImpurityDecrease = 0
        } else {
            self.minImpurityDecrease = minImpurityDecrease
        }
        self.scoring = scoring
    }
}
