import TensorFlow

// typealias IntValue = Int32

protocol TreeEstimator: Estimator {
    var featureImportances: Tensor<Float> { get }
    // var scoring: String { get }

    // mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>)

    // func predict(data x: Tensor<Float>) -> Tensor<Float>

    // func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float
}

/// A criterion function takes input of groups of labels (either classes or
/// numerical values) in the form of array of 2 arraies [left, right]
typealias CriterionFn = ([[Float]]) -> Float

// func giniImpurity(_ groups: [[Float]], _ classes: [Float]) -> Float {
func giniImpurity(_ groups: [[Float]]) -> Float {
    let classes = groups.joined()
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

// func mseCriterion(_ groups: [[Float]], _ classes: [Float]) -> Float {
func mseCriterion(_ groups: [[Float]]) -> Float {
    var N: Float = 0.0
    for group in groups {
        N += Float(group.count)
    }

    var err: Float = 0

    for group in groups {
        let sizeG = Float(group.count)
        if sizeG == 0 { continue }
        let mean = group.reduce(0, +) / sizeG
        print("group: \(group)")
        print("mean: \(mean)")
        // use a tensor fill with mean
        let y = Tensor<Float>(repeating: mean, shape: [Int(sizeG), 1])
        let yhat = Tensor<Float>(group)
        err += meanSquaredErrorTF(y, yhat)
    }
    return err
}

/// can use a dict for now, change to function with switch for more complcated cases
let Criteria = [
    "gini": giniImpurity,
    "mse": mseCriterion,
]

typealias Groups = (left: [Int], right: [Int])
typealias LabelGroups = (left: [Float], right: [Float])

class Node: CustomStringConvertible {
    var id: Int
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

    init(id: Int, depth: Int, feature: Int, splitValue: Float, score: Float, groups: Groups) {
        self.isEmpty = false
        self.id = id
        self.depth = depth
        self.feature = feature
        self.splitValue = splitValue
        self.score = score
        self.groups = groups
    }

    init(isEmpty: Bool) {
        self.isEmpty = true
        self.id = -1
        self.depth = -1
        self.feature = -1
        self.splitValue = -1
        self.score = -1
        self.groups = (left: [Int](), right: [Int]())
    }

    public var description: String {
        if isEmpty { return "[empty node]" }

        let space = String(repeating: " ", count: depth * 2)
        var desc = "\(space)id: \(id), leaf: \(isLeaf), "

        // let possibleValue = value ?? 0.0
        if isLeaf { desc += "value: \(value!), " }

        return desc + """
        feature: \(feature), splitValue: \(splitValue), score: \(score)
        """
    }
}

/// The root N of T is stored in TREE [1].
/// If a node occupies TREE [k] then its left child is stored in TREE [2 * k]
/// and its right child is stored into TREE [2 * k + 1].
class DTree {
    var nodes: [Node]
    init() { nodes = [Node]() }
    func addNode(_ node: Node) {
        // nodes.append(node)
        // print("nodes count: \(nodes.count)")
        // print("nodes cap: \(nodes.capacity)")
        // print("adding node: \(node.id)")
        if (node.id - 1) >= nodes.count {
            // nodes.reserveCapacity(node.id * 2 + 1)
            paddingEmptyNodes(node.id * 2 + 1)
        }

        // print("nodes count: \(nodes.count)")
        nodes.insert(node, at: node.id - 1)
    }

    func addLeftChild(parent: Node, child: Node) {
        child.id = parent.id * 2
        parent.leftChild = child.id
        nodes.insert(child, at: parent.id * 2 - 1)
    }

    func addRightChild(parent: Node, child: Node) {
        child.id = parent.id * 2 + 1
        parent.rightChild = child.id
        nodes.insert(child, at: parent.id * 2)
    }

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
                if x[i, node.feature].scalar! < node.splitValue {
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
        return nodes[parent.id * 2 - 1]
    }

    func rightChild(_ parent: Node) -> Node {
        return nodes[parent.id * 2]
    }
}

struct BestFirstTreeBuilder {
    let criterion: CriterionFn
    let isClassification: Bool
    var nFeatures: Int
    var classes: [Int]
    var nOutputs: Int
    var maxDepth: Int
    var minSamplesSplit: Int
    var maxFeatures: Int
    // var criterionFn: CriterionFn

    func build(data: Matrix) -> DTree {
        let tree = DTree()
        let depth = 0
        let node = addSplitNode(data: data, depth: depth, isFirst: true, isLeft: nil, parent: nil)
        tree.addNode(node)

        splitNode(tree: tree, node: node, data: data, depth: depth + 1)

        return tree
    }

    func splitNode(tree: DTree, node: Node, data: Matrix, depth: Int) {
        if isLeaf(node) {
            // print("rearched Leaf")
            markLeaf(node, data: data)
            return
        }
        let (left, right) = node.groups
        // let leftData = dataSample(idx: left, data: data)
        let leftData = data.select(rows: left)
        let leftNode = addSplitNode(data: leftData, depth: depth, isFirst: false, isLeft: true, parent: node)
        // tree.addLeftChild(parent: node, child: leftNode)
        tree.addNode(leftNode)

        // let rightData = dataSample(idx: right, data: data)
        let rightData = data.select(rows: right)
        let rightNode = addSplitNode(data: rightData, depth: depth, isFirst: false, isLeft: false, parent: node)
        // tree.addRightChild(parent: node, child: rightNode)
        tree.addNode(rightNode)

        splitNode(tree: tree, node: leftNode, data: leftData, depth: depth + 1)
        splitNode(tree: tree, node: rightNode, data: rightData, depth: depth + 1)
    }

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

        // if node.groups.left == nil || node.groups.right == nil {
        if node.groups.left.count == 0 || node.groups.right.count == 0 {
            return true
        }
        if node.depth >= maxDepth {
            return true
        }
        if node.groups.left.count < minSamplesSplit || node.groups.right.count < minSamplesSplit {
            return true
        }
        return false
    }

    func addSplitNode(data: Matrix, depth: Int, isFirst: Bool, isLeft: Bool?, parent: Node?) -> Node {
        var bstScore: Float = 1
        var bstCol: Int = 0
        var bstSplitValue: Float = -1
        var bstGroups: Groups?
        var nodeId: Int = 0

        for col in 0 ..< nFeatures - 1 {
            for value in data[0..., col].scalars {
                let sampleSplit = getSampleSplit(col: col, splitBy: value, data: data)
                // print("sampleSplit: \(sampleSplit)")
                let labelGroups = getLabelGroups(sampleSplit: sampleSplit, data: data)
                // print("labelGroups: \(labelGroups)")
                let score = criterion([labelGroups.left, labelGroups.right])
                // print("score: \(score)")
                if score < bstScore {
                    bstScore = score
                    bstCol = col
                    bstSplitValue = value
                    // bstGroups = groups
                    bstGroups = sampleSplit
                }
            }
        }
        if isFirst {
            nodeId = 1
        } else if parent != nil, isLeft != nil {
            if isLeft! {
                nodeId = parent!.id * 2
            } else {
                nodeId = parent!.id * 2 + 1
            }
        } else {
            print("Should provide isLeft and parent node if not first!")
        }
        // print("bstGroups: \(bstGroups)")
        return Node(id: nodeId, depth: depth, feature: bstCol, splitValue: bstSplitValue, score: bstScore,
                    groups: bstGroups!)
    }

    func getSampleSplit(col: Int, splitBy: Float, data: Matrix) -> Groups {
        var left = [Int](), right = [Int]()
        for (idx, value) in data[0..., col].scalars.enumerated() {
            let rowIdx = Int(idx)

            if value < splitBy {
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

struct DecisionTree: TreeEstimator {
    var criterion, splitter: String
    var nFeatures: Int = 0
    var maxDepth: Int
    var maxFeatures: Int
    var minSamplesSplit: Int
    var nClasses: Int = 0
    var tree: DTree?
    var featureImportances: Tensor<Float> { return Tensor(0) }
    var scoring: String = "accuracy"

    init(criterion: String = "gini", splitter: String = "best",
         maxDepth: Int = -1,
         maxFeatures: Int = -1, minSamplesSplit: Int = 1,
         scoring: String = "accuracy") {
        (self.criterion, self.splitter) = (criterion, splitter)
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.minSamplesSplit = minSamplesSplit
        self.scoring = scoring
    }

    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>) {
        //// check input data is 2d
        print(x.shape)

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
            maxDepth = 9999
        }

        if maxFeatures == -1 || maxFeatures > nFeatures {
            maxFeatures = nFeatures
        }

        guard let fnCriterion = Criteria[criterion] else {
            print("criterion not found")
            return
        }
        let isClassification = true

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
                                           maxFeatures: maxFeatures)

        tree = builder.build(data: data)
        // print(tree.score)
    }

    func predict(data x: Tensor<Float>) -> Tensor<Float> {
        let proba = tree!.predict(x)
        // print(proba)
        let result = Tensor<Float>(proba)
        // print(type(of:result))
        // print(result.shape)
        return result
    }

    func printTree() {
        guard self.tree != nil else {
            print("Tree not built!")
            return
        }
        let tree = self.tree!
        // printNode(node: node, depth: 0)
        // print(tree.nodes)
        for node in tree.nodes {
            if node.isEmpty { continue }
            print(node)
        }
    }
}
