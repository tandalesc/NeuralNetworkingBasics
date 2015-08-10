using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkingBasics
{
    class Network
    {
        private int[] nodesInEachLayer;
        private double[][] biases;
        private double[][][] weights;
        private System.Random rndm = new System.Random();
        private int NumberOfLayers
        {
            get
            {
                return nodesInEachLayer.Length;
            }
        }

        public Network(params int[] nodes)
        {
            nodesInEachLayer = nodes;

            biases = new double[NumberOfLayers][];
            weights = new double[NumberOfLayers][][];

            SetUpBiasesAndWeights();
        }

        //Learning methods
        public void GradientDescent(List<double[]> inputs, List<double[]> expectedOutputs, int iterations, int batchSize, double learningRate, double regularizationFactor)
        {
            for (int iteration = 0; iteration < iterations; iteration++)
            {

                int[] indexesToRemove = GetRandomIndexes(inputs, batchSize);
                double[][] inputSet = GetBatch(inputs, indexesToRemove);
                double[][] outputSet = GetBatch(expectedOutputs, indexesToRemove);

                UpdateBiasesAndWeights(inputSet, outputSet, learningRate, batchSize, regularizationFactor);
            }
        }
        private Errors BackPropagate(double[] inputs, double[] expectedOutputs, double regularizationFactor, int totalInputs)
        {
            //set up
            double[] realOutputs = FeedForward(inputs);
            double[][] errorInBiases = new double[NumberOfLayers][];
            double[][][] errorInWeights = this.weights;

            int lastLayer = NumberOfLayers - 1;
            int numberOfOutputNodes = nodesInEachLayer[lastLayer];

            //get errors in biases (or just errors in general -- same thing)
            double[] difference = MatrixAdd(realOutputs, ConstantMultiply(-1, expectedOutputs));
            //errorInBiases[lastLayer] = HadamardProduct(difference, SigmoidPrime(CalculateZ(lastLayer)));
            errorInBiases[lastLayer] = difference; //cross-entropy

            //back propagate errors
            for (int layer = lastLayer - 1; layer >= 0; layer--)
            {
                errorInBiases[layer] = new double[nodesInEachLayer[layer]];


                double[][] wT = Transpose(weights[layer + 1]);
                double[] propagatedError = MatrixMultiply(wT, errorInBiases[layer + 1]);
                errorInBiases[layer] = HadamardProduct(propagatedError, SigmoidPrime(CalculateZ(layer)));
            }

            //get error in weights
            for (int layer = 0; layer < NumberOfLayers; layer++)
            {
                for (int node = 0; node < nodesInEachLayer[layer]; node++)
                {
                    if (layer > 0)
                    {
                        for (int inputNode = 0; inputNode < nodesInEachLayer[layer - 1]; inputNode++)
                        {
                            errorInWeights[layer][node][inputNode] = SigmoidPrime(CalculateZ(layer - 1, inputNode)) * errorInBiases[layer][node];
                            errorInWeights[layer][node][inputNode] += regularizationFactor / totalInputs * this.weights[layer][node][inputNode];
                        }
                    }
                    else
                    {
                        errorInWeights[0][node] = this.weights[0][node];
                    }
                }
            }
            errorInWeights[0] = this.weights[0];

            //package errors and return
            Errors e = new Errors();
            e.partial_B = errorInBiases;
            e.partial_W = errorInWeights;

            return e;
        }
        private void UpdateBiasesAndWeights(double[][] inputSet, double[][] outputSet, double learningRate, int batchSize, double regularizationFactor)
        {
            double[][] delta_b_t = new double[NumberOfLayers][];
            double[][][] delta_w_t = new double[NumberOfLayers][][];

            for (int i = 0; i < inputSet.Length; i++)
            {
                Errors e = BackPropagate(inputSet[i], outputSet[i], regularizationFactor, inputSet.Length);

                double[][] delta_b = MatrixAdd(e.partial_B, this.biases);
                double[][][] delta_w = MatrixAdd(e.partial_W, this.weights);

                if (i == 0) //instantiate if necessary
                {
                    delta_b_t = delta_b;
                    delta_w_t = delta_w;
                }

                delta_b_t = MatrixAdd(delta_b_t, delta_b);
                delta_w_t = MatrixAdd(delta_w_t, delta_w);
            }

            for (int layer = 0; layer < NumberOfLayers; layer++)
            {
                this.biases[layer] = MatrixAdd(this.biases[layer], ConstantMultiply(-1 * learningRate / batchSize, delta_b_t[layer]));

                for (int node = 0; node < nodesInEachLayer[layer]; node++)
                {
                    this.weights[layer][node] = MatrixAdd(ConstantMultiply(1 - learningRate*regularizationFactor/inputSet.Length, this.weights[layer][node])
                        , ConstantMultiply(-1 * learningRate / batchSize, delta_w_t[layer][node]));
                }
            }
        }
        public double[] Run(params double[] inputs) { return FeedForward(inputs); }

        //Network Utility methods
        private double[] FeedForward(double[] inputs)
        {
            biases[0] = inputs; //update biases

            double[] outputs = Sigmoid(CalculateZ(NumberOfLayers - 1));

            return outputs;
        }
        private double[][] FeedForward(double[][] inputs)
        {
            double[][] outputs = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                biases[0] = inputs[i]; //update biases

                outputs[i] = Sigmoid(CalculateZ(NumberOfLayers - 1));
            }
            return outputs;
        }
        private double CalculateZ(int layer, int node)
        {
            double weightedSum = 0.0;

            if (layer > 0)
                for (int input = 0; input < nodesInEachLayer[layer - 1]; input++)
                {
                    weightedSum += weights[layer][node][input] * CalculateZ(layer - 1, input);
                }

            return weightedSum + this.biases[layer][node];
        }
        private double[] CalculateZ(int layer)
        {
            double[] zs = new double[nodesInEachLayer[layer]];
            for (int i = 0; i < zs.Length; i++)
            {
                zs[i] = CalculateZ(layer, i);
            }
            return zs;
        }
        private void SetUpBiasesAndWeights()
        {
            for (int layer = 0; layer < NumberOfLayers; layer++)
            {
                int numberOfNodesInLayer = nodesInEachLayer[layer];

                this.biases[layer] = new double[numberOfNodesInLayer];
                this.weights[layer] = new double[numberOfNodesInLayer][];

                for (int node = 0; node < numberOfNodesInLayer; node++)
                {
                    biases[layer][node] = RandomNum(-1, 1);

                    if (layer > 0)
                    {
                        weights[layer][node] = new double[nodesInEachLayer[layer - 1]];
                        for (int previousNode = 0; previousNode < nodesInEachLayer[layer - 1]; previousNode++)
                        {
                            weights[layer][node][previousNode] = RandomNum(-1, 1) / Math.Sqrt(nodesInEachLayer[layer - 1]);
                        }
                    }
                    else
                        weights[0][node] = new double[] { };
                }
            }

            // :)
        }
        private double Sigmoid(double x)
        {
            double diff = 1 + Math.Exp(-1 * x);
            return 1.0 / diff;
        }
        private double[] Sigmoid(double[] xs)
        {
            double[] sp = new double[xs.Length];
            for (int i = 0; i < sp.Length; i++)
            {
                sp[i] = Sigmoid(xs[i]);
            }
            return sp;
        }
        private double SigmoidPrime(double x)
        {
            double s = Sigmoid(x);
            return s * (1 - s);
        }
        private double[] SigmoidPrime(double[] xs)
        {
            double[] sp = new double[xs.Length];
            for (int i = 0; i < sp.Length; i++)
            {
                sp[i] = SigmoidPrime(xs[i]);
            }
            return sp;
        }
        private double RandomNum(double min, double max)
        {
            double range = max - min;
            double interval = rndm.NextDouble() * (range); // (0, max - min);
            return interval + min; //(min, max)
        }

        //Matrix methods
        private double[] HadamardProduct(double[] a, double[] b)
        {
            double[] atimesb = new double[a.Length];
            for (int i = 0; i < atimesb.Length; i++)
                atimesb[i] = a[i] * b[i];
            return atimesb;
        }
        private double[] ConstantMultiply(double a, double[] b)
        {
            double[] atimesb = new double[b.Length];
            for (int i = 0; i < atimesb.Length; i++)
                atimesb[i] = a * b[i];
            return atimesb;
        }
        private double[][] ConstantMultiply(double a, double[][] b)
        {
            double[][] atimesb = new double[b.Length][];
            for (int i = 0; i < atimesb.Length; i++)
                atimesb[i] = ConstantMultiply(a,b[i]);
            return atimesb;
        }
        private double[][] Transpose(double[][] a)
        {
            int rows = a.Length;
            int columns = a[0].Length;

            double[][] aT = new double[columns][];
            for (int c = 0; c < columns; c++) 
            {
                aT[c] = new double[rows];
                for (int r = 0; r < rows; r++)
                {
                    aT[c][r] = a[r][c];
                }
            }

            return aT;
        }
        private double[] MatrixMultiply(double[][] a, double[] v)
        {
            double[] resultant = new double[a.Length];

            for (int row = 0; row < a.Length; row++)
            {
                resultant[row] = 0.0;
                for (int vindex = 0; vindex < v.Length; vindex++)
                {
                    resultant[row] += a[row][vindex] * v[vindex];
                }
            }

            return resultant;
        }
        private double[] MatrixAdd(double[] a, double[] b)
        {
            double[] ab = new double[a.Length];
            for (int i = 0; i < ab.Length; i++)
                ab[i] = a[i] + b[i];
            return ab;
        }
        private double[][] MatrixAdd(double[][] a, double[][] b)
        {
            double[][] ab = new double[a.Length][];
            for (int i = 0; i < a.Length; i++)
                ab[i] = MatrixAdd(a[i], b[i]);
            return ab;
        }
        private double[][][] MatrixAdd(double[][][] a, double[][][] b)
        {
            double[][][] ab = new double[a.Length][][];
            for (int i = 0; i < a.Length; i++)
                ab[i] = MatrixAdd(a[i], b[i]);
            return ab;
        }

        //List methods
        private List<double[]> GetSubset(List<double[]> l, int[] elems)
        {
            List<double[]> rets = new List<double[]>();
            foreach (int i in elems)
            {
                rets.Add(l[i]);
            }
            return rets;
        }
        private double[][] GetBatch(List<double[]> l, int[] elems) 
        {
            return GetSubset(l, elems).ToArray();
        }
        private int[] GetRandomIndexes(List<double[]> l, int count)
        {
            List<int> indexes = new List<int>();

            while (indexes.Count < count)
            {
                int j = rndm.Next(l.Count);
                if (j < count)
                {
                    if(!indexes.Contains(j))
                        indexes.Add(j);
                }
            }

            return indexes.ToArray();
        }
    }
    struct Errors
    {
        public double[][] partial_B;
        public double[][][] partial_W;
    }
}
