using System.Data;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Xml.Linq;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace DecisionTree
{
    public class DataFrame
    {
        public DataFrame(List<List<float>> x, List<int> y)
        {
            X = x;
            Y = y;
        }

        public List<List<float>> X { get; set; } //features variables
        public List<int> Y { get; set; } //target variables
        
    }
    public class TreeNode
    {
        public TreeNode(int featureIndex, float treshold, TreeNode? leftNode, TreeNode? righNode, float infoGain,float? leafValue)
        {
            FeatureIndex = featureIndex;
            Treshold = treshold;  
            LeftNode = leftNode;
            RightNode = righNode;
            InfoGain = infoGain;
            LeafValue = leafValue;
        }
        // as DecisionTree is just an bunch of nestest if statements
        #region "if statemnet" part of tree
        // example: left node -> x<treshold; right node x>=treshold
        // where x is an DataFrame.X[FeatureIndex]
        public int FeatureIndex { get; set; } //index of feature used for split
        public float Treshold { get; set; }// value witch splits feachuers to splits samples to left and right notde

        public TreeNode? LeftNode { get; set; }
        public TreeNode? RightNode { get; set; }
        #endregion

        public float InfoGain { get; set; }
        public float? LeafValue { get; set; }
    }
    public class BestSplit
    {
        public BestSplit(DataFrame leftDataFrame, DataFrame rightDataFrame, int splitingFeatureIndex, float tresholdValue, float infoGain)
        {
            this.LeftDataFrame = leftDataFrame;
            this.RightDataFrame = rightDataFrame;
            this.SplitingFeatureIndex = splitingFeatureIndex;
            this.TresholdValue = tresholdValue;
            this.InfoGain = infoGain;
        }
        public DataFrame LeftDataFrame { get; set; }
        public DataFrame RightDataFrame { get; set; }
        public int SplitingFeatureIndex { get; set; }
        public float TresholdValue {  get; set; }
        public float InfoGain { get; set; }
    }
    public class Metrics
    {
        public double Accuracy;
        public double Precision;
        public double Recall;
        public double F1Score;
    }
    public class DecisionTreeClassifier()
    {
        public List<int> Predicate(TreeNode trainedTreeNode, List<List<float>> samplesToPredict)
        {
            List<int> yPredictions = new List<int>();
            for (int sampleIndex = 0; sampleIndex < samplesToPredict.Count; sampleIndex++)
            {
                int? yPrediction = PredicateInstance(trainedTreeNode, samplesToPredict[sampleIndex]);
                if (yPrediction != null) { yPredictions.Add((int)yPrediction); }
            }
            return yPredictions;
        }

        public int? PredicateInstance(TreeNode trainedTreeNode,List<float> sampleToPredict)
        {
            //stop conditon
            if (trainedTreeNode.LeftNode == null && trainedTreeNode.RightNode == null)
            {
                return (int?)trainedTreeNode.LeafValue;
            }
            //The ! (null-forgiving operator) tells the C# compiler: "I know this value will never be null, so stop warning me about possible null references."
            //Regardles of above will check for null nodes even if logically they shouldn't be. Oceurence of such inidcates wrongly build tree.
            if (sampleToPredict[trainedTreeNode.FeatureIndex] < trainedTreeNode.Treshold)
            {
                if (trainedTreeNode.LeftNode == null)
                    throw new InvalidOperationException("LeftNode is null in a non-leaf node. Worth checking if spliting was done correctly during tree training/building proces");

                return PredicateInstance(trainedTreeNode.LeftNode, sampleToPredict);
            }
            else
            {
                if (trainedTreeNode.RightNode == null)
                    throw new InvalidOperationException("RightNode is null in a non-leaf node. Worth checking if spliting was done correctly during tree training/building proces");

                return PredicateInstance(trainedTreeNode.RightNode, sampleToPredict);
            }
        }
        //implemet voting system insted of taking first value as an leafValue curent case works if leafNodes are 100% pure
        public TreeNode BuildTree(DataFrame dataFrame, int maxTreeHeight, int currentTreeHight = 0)
        {
            BestSplit bestSplit = GetBestSplit(dataFrame);
            //stop condition
            if (bestSplit.InfoGain <= 0 || currentTreeHight >= maxTreeHeight)
            {
                return new TreeNode(bestSplit.SplitingFeatureIndex, bestSplit.TresholdValue, null, null, bestSplit.InfoGain, dataFrame.Y[0]);
            }
            else
            {
                currentTreeHight++;
                TreeNode leftNode =  BuildTree(bestSplit.LeftDataFrame, maxTreeHeight, currentTreeHight);
                TreeNode rightNode = BuildTree(bestSplit.RightDataFrame, maxTreeHeight, currentTreeHight);
                return new TreeNode(bestSplit.SplitingFeatureIndex, bestSplit.TresholdValue, leftNode, rightNode, bestSplit.InfoGain, null);
            }
        }

        public BestSplit GetBestSplit(DataFrame dataFrame)
        {
            BestSplit bestSplit = new BestSplit(dataFrame, dataFrame, 0, 0, 0);
            for (int sampleIndex = 0; sampleIndex < dataFrame.X.Count; sampleIndex++)
            {
                for (int featuresIndex = 0; featuresIndex < dataFrame.X[sampleIndex].Count; featuresIndex++)
                {
                    var splitedDataFrame = Split(dataFrame, dataFrame.X[sampleIndex][featuresIndex], featuresIndex);
                    float infoGain = CalculateInfoGain(dataFrame, splitedDataFrame.LeftDataFrame, splitedDataFrame.RightDataFrame);
                    if(infoGain > bestSplit.InfoGain) 
                    {
                        bestSplit = new BestSplit(splitedDataFrame.LeftDataFrame, splitedDataFrame.RightDataFrame, featuresIndex, dataFrame.X[sampleIndex][featuresIndex], infoGain); 
                    }
                }
            }
            return bestSplit;
        }
        public (DataFrame LeftDataFrame, DataFrame RightDataFrame) Split(DataFrame dataFrameToSplit, float tresholdValue, int splitingFeatureIndex) //tresholdValue = X[chosenSample][splitingFeatureIndex]
        {
            DataFrame leftDataFrame = new DataFrame(new List<List<float>>(), new List<int>());
            DataFrame rightDataFrame = new DataFrame(new List<List<float>>(), new List<int>());

            for (int sampleIndex = 0; sampleIndex<dataFrameToSplit.X.Count; sampleIndex++)
            {
                if (dataFrameToSplit.X[sampleIndex][splitingFeatureIndex] < tresholdValue)
                {
                    leftDataFrame.X.Add(dataFrameToSplit.X[sampleIndex]);
                    leftDataFrame.Y.Add(dataFrameToSplit.Y[sampleIndex]);
                }
                else
                {
                    rightDataFrame.X.Add(dataFrameToSplit.X[sampleIndex]);
                    rightDataFrame.Y.Add(dataFrameToSplit.Y[sampleIndex]);
                }
            }
            return (leftDataFrame, rightDataFrame);
        }

        public float CalculateInfoGain(DataFrame parenDataFrame, DataFrame leftDataFrame, DataFrame rightDataFrame)
        {
            int numOfParentTargetVariables = parenDataFrame.Y.Count;

            double parentEntropy = CalculateEntropy(parenDataFrame.Y);

            double leftChildEntropy = CalculateEntropy(leftDataFrame.Y);
            double leftChildWeight = (double)leftDataFrame.Y.Count / numOfParentTargetVariables;

            double rightChildEntropy = CalculateEntropy(rightDataFrame.Y);
            double rightChildWeight = (double)rightDataFrame.Y.Count / numOfParentTargetVariables;

            double infoGain = parentEntropy - ((leftChildWeight * leftChildEntropy) + (rightChildWeight * rightChildEntropy));
            return (float)infoGain;
        }
        /// <summary>
        /// Computes the entropy of the target variables classified to this specific node.
        /// Entropy quantifies the uncertainty or impurity within the node.
        /// </summary>
        /// <param name="y">An array of target variables that have been classified to this node.</param>
        /// <returns>The entropy value as a double.</returns>
        public double CalculateEntropy(List<int> y)
        {
            int[] distinctClasses = y.Distinct().ToArray();
            double entropy = 0;
            for (int i = 0; i < distinctClasses.Length; i++)
            {
                double iPropabilty = (double)y.Count(x =>x == distinctClasses[i]) / y.Count;
                entropy += (iPropabilty * Math.Log(iPropabilty));
            }
            return -entropy;
        }
    }

    public class Helper
    {

        public (float[,], int[]) LoadCSVToArrayTuple(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath);

            /* int featuresNum = lines[0]?.Split(',').Length?? 0; Will be save in case whole array is null or Lenght = 0
             * Solution:
             * int featuresNum = 0;
             * if(lines!=null || lines.Lenght>=0) {featuresNum = lines[0].split(',').Lenght;}
             * Used aproach below is using LINQ so it can be an one liner. */
            //if first line in lines is null or empty set featuresNum to 0. For List (in consequence arrays to) FirstOrDefault has time complexity of O(1)
            int featuresNum = lines.FirstOrDefault()?.Split(',').Length ?? 0; 

            float[,] x = new float[lines.Length, featuresNum-1];
            int[] y = new int[lines.Length];

            for(int sampleIndex=1; sampleIndex<lines.Length; sampleIndex++)
            {
                string[] feachersInLine = lines[sampleIndex].Split(",", featuresNum);
                for (int featuresIndex = 0; featuresIndex < featuresNum-1; featuresIndex++)
                {
                    x[sampleIndex,featuresIndex] = float.Parse(feachersInLine[featuresIndex]);
                }
                y[sampleIndex] = Int32.Parse(feachersInLine[featuresNum-1]);
            }
            return (x, y);
        }

        public DataFrame LoadCSVToDataFrame(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath);

            int featuresNum = lines.FirstOrDefault()?.Split(',').Length ?? 0;

            List<List<float>> x = new List<List<float>>();
            List<int> y = new List<int>();

            for(int sampleIndex=1; sampleIndex<lines.Length; sampleIndex++)
            {
                string[] feachersInLine = lines[sampleIndex].Split(",");
                List<float> featuers = new List<float>();
                for(int featuresIndex = 0; featuresIndex< featuresNum-1; featuresIndex++)
                {
                    featuers.Add(float.Parse(feachersInLine[featuresIndex]));
                }
                x.Add(featuers);
                y.Add(Int32.Parse(feachersInLine[featuresNum - 1]));
            }

            return new DataFrame(x, y);
        }
        public void PrintDataFrame(DataFrame dataFrame, int? topRowsToPrint = null)
        {
            if (topRowsToPrint==null || topRowsToPrint > dataFrame.X.Count) { topRowsToPrint = dataFrame.X.Count; }

            for (int samples = 0; samples < topRowsToPrint; samples++)
            {
                for (int features = 0; features < dataFrame.X[samples].Count; features++)
                {
                    Console.Write(dataFrame.X[samples][features] + ", ");
                }
                Console.WriteLine(dataFrame.Y[samples]);
            }
        }
        /// <summary>
        /// Randomizes the order of the data within the given DataFrame by shuffling its elements in place.
        /// This method modifies the original DataFrame rather than creating a new one.
        /// </summary>
        /// <param name="dataFrame">The DataFrame to be randomized. Its contents will be shuffled in place.</param>
        /// <param name="randomSeed">The seed for the random number generator to ensure reproducibility.</param>
        public void RandomizeDataFrame(DataFrame dataFrame, int randomSeed)
        {
            Random random = new Random(randomSeed);
            for(int sampleIndex = 0; sampleIndex < dataFrame.X.Count; sampleIndex++)
            {
                int randomSampleIndex = random.Next(dataFrame.X.Count);
                SwapListElements(dataFrame.X, sampleIndex, randomSampleIndex);
                SwapListElements(dataFrame.Y, sampleIndex, randomSampleIndex);
            }
        }
        // could have used: list.GetRange(0, trainCount); insted of while loops
        public (DataFrame train, DataFrame test) TrainTestDataFrameSplit(DataFrame dataFrameToSpilt, float testProcentSize = 0.2f) 
        {
            DataFrame train = new DataFrame(new List<List<float>>(),new List<int>());
            DataFrame test = new DataFrame(new List<List<float>>(), new List<int>());

            int maxTrainIndex = (int)((1-testProcentSize)*dataFrameToSpilt.X.Count);
            int sampleIndex = 0;
            while (sampleIndex < maxTrainIndex)
            {
                train.X.Add(dataFrameToSpilt.X[sampleIndex]);
                train.Y.Add(dataFrameToSpilt.Y[sampleIndex]);
                sampleIndex++;
            }
            while (sampleIndex < dataFrameToSpilt.X.Count)
            {
                test.X.Add(dataFrameToSpilt.X[sampleIndex]);
                test.Y.Add(dataFrameToSpilt.Y[sampleIndex]);
                sampleIndex++;
            }
            return (train, test);
        }

        public void SwapListElements<T>(List<T> list, int index1, int index2)
        {
            T temp = list[index1];
            list[index1] = list[index2];
            list[index2] = temp;
        }

        //This version of EvaluateClassificationModel supports only classification with 2 classes (example: yes and no class)
        public Metrics EvaluateClassificationModel(List<int> expected, List<int> predicted)
        {
            if (expected.Count != predicted.Count || expected.Count == 0)
                throw new ArgumentException("Lists must be the same length and cannot be empty.");

            // Calculate mistake matrix
            int TP = expected.Zip(predicted, (e, p) => e == 1 && p == 1 ? 1 : 0).Sum();
            int TN = expected.Zip(predicted, (e, p) => e == 0 && p == 0 ? 1 : 0).Sum();
            int FP = expected.Zip(predicted, (e, p) => e == 0 && p == 1 ? 1 : 0).Sum();
            int FN = expected.Zip(predicted, (e, p) => e == 1 && p == 0 ? 1 : 0).Sum();

            Metrics result = new Metrics();

            // Calculate metrics
            int total = expected.Count;
            result.Accuracy = (double)(TP + TN) / total;
            result.Precision = (TP + FP) > 0 ? (double)TP / (TP + FP) : 0;
            result.Recall = (TP + FN) > 0 ? (double)TP / (TP + FN) : 0;
            result.F1Score = (result.Precision + result.Recall) > 0
                ? 2 * (result.Precision * result.Recall) / (result.Precision + result.Recall)
                : 0;

            return result;
        }
    }
    internal class Program
    {
        static void Main(string[] args)
        {
            string filePath = @"file path to data csv";
            Helper helper = new Helper();
            DataFrame dataFrame = helper.LoadCSVToDataFrame(filePath);
            
            Console.WriteLine("Top 10 elements of org DataFrame:");
            helper.PrintDataFrame(dataFrame,10);
            Console.WriteLine("Top 10 elements of randomized DataFrame:");
            helper.RandomizeDataFrame(dataFrame, 313012);
            helper.PrintDataFrame(dataFrame, 10);

            var trainTestDataFrames = helper.TrainTestDataFrameSplit(dataFrame);

            DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier();
            
            TreeNode root = decisionTreeClassifier.BuildTree(trainTestDataFrames.train, 10);

            //List<float> testSample = new List<float> { 13.54f, 14.36f, 87.46f, 566.3f, 0.09779f };
            List<int> predictedYs = decisionTreeClassifier.Predicate(root, trainTestDataFrames.test.X);

            Console.WriteLine("Train size : " + trainTestDataFrames.train.Y.Count + ", test size : " + trainTestDataFrames.test.Y.Count);

            Metrics metrics = helper.EvaluateClassificationModel(trainTestDataFrames.test.Y, predictedYs);
            Console.WriteLine("Accuracy: " + metrics.Accuracy.ToString());
            Console.WriteLine("Precision: " + metrics.Precision.ToString());
            Console.WriteLine("Recall: " + metrics.Recall.ToString());
            Console.WriteLine("F1Score: " + metrics.F1Score.ToString());
        }
    }
}
