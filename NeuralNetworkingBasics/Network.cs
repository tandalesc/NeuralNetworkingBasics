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
        private double[][][] velocity;
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
            velocity = new double[NumberOfLayers][][];

            SetUpBiasesAndWeights();
        }

        //Learning methods
        public void GradientDescent(List<double[]> inputs, List<double[]> expectedOutputs, int iterations, int batchSize, double learningRate, double regularizationFactor, double friction)
        {
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                int[] indexesToRemove = GetRandomIndexes(inputs, batchSize);
                double[][] inputSet = GetBatch(inputs, indexesToRemove);
                double[][] outputSet = GetBatch(expectedOutputs, indexesToRemove);

                UpdateBiasesAndWeights(inputSet, outputSet, learningRate, batchSize, regularizationFactor, friction);
            }
        }
        private Errors BackPropagate(double[] inputs, double[] expectedOutputs, double regularizationFactor, int totalInputs)
        {
            //set up
            double[] realOutputs = FeedForward(inputs);
            double[][] errorInBiases = ConstantMultiply(0, this.biases);
            double[][][] errorInWeights = ConstantMultiply(0, this.weights);

            int lastLayer = NumberOfLayers - 1;
            int numberOfOutputNodes = nodesInEachLayer[lastLayer];

            //get errors in biases (or just errors in general -- same thing)
            double[] difference = MatrixAdd(realOutputs, ConstantMultiply(-1, expectedOutputs));
            //errorInBiases[lastLayer] = HadamardProduct(difference, SigmoidPrime(CalculateZ(lastLayer))); //parabolic
            errorInBiases[lastLayer] = difference; //cross-entropy

            //back propagate errors
            for (int layer = lastLayer - 1; layer >= 0; layer--)
            {
                errorInBiases[layer] = ConstantMultiply(0, this.biases[layer]);


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
                            //errorInWeights[layer][node][inputNode] += regularizationFactor / totalInputs * this.weights[layer][node][inputNode];
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
        private void UpdateBiasesAndWeights(double[][] inputSet, double[][] outputSet, double learningRate, int batchSize, double regularizationFactor, double friction)
        {
            double[][] delta_b_t = ConstantMultiply(0, this.biases);
            double[][][] delta_w_t = ConstantMultiply(0, this.weights);

            for (int i = 0; i < inputSet.Length; i++)
            {
                Errors e = BackPropagate(inputSet[i], outputSet[i], regularizationFactor, inputSet.Length);

                delta_b_t = MatrixAdd(delta_b_t, e.partial_B);
                delta_w_t = MatrixAdd(delta_w_t, e.partial_W);
            }

            for (int layer = 1; layer < NumberOfLayers; layer++)
            {
                this.biases[layer] = MatrixAdd(this.biases[layer], ConstantMultiply(-1 * learningRate / batchSize, delta_b_t[layer]));

                for (int node = 0; node < nodesInEachLayer[layer]; node++)
                {
                    //momentum-based stochastic, regulated gradient descent
                    this.velocity[layer][node] = MatrixAdd(ConstantMultiply(friction /* - learningRate*regularizationFactor*friction/inputSet.Length*/, this.velocity[layer][node])
                        , ConstantMultiply(-1 * learningRate / batchSize, delta_w_t[layer][node]));

                    this.weights[layer][node] = MatrixAdd(this.weights[layer][node], this.velocity[layer][node]);
                }
            }
        }
        public double[] Run(params double[] inputs) { return FeedForwardIgnoreZero(inputs); }

        //Network Utility methods
        private double[] FeedForward(double[] inputs)
        {
            biases[0] = inputs; //update biases

            double[] outputs = Sigmoid(CalculateZ(NumberOfLayers - 1));

            return outputs;
        }
        private double[] FeedForwardIgnoreZero(double[] inputs)
        {
            biases[0] = inputs; //update biases

            double[] outputs = SigmoidIgnoreZero(CalculateZ(NumberOfLayers - 1));

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
        private double[][] FeedForwardIgnoreZero(double[][] inputs)
        {
            double[][] outputs = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                biases[0] = inputs[i]; //update biases

                outputs[i] = SigmoidIgnoreZero(CalculateZ(NumberOfLayers - 1));
            }
            return outputs;
        }
        private double CalculateZ(int layer, int node)
        {
            double weightedSum = 0.0;

            if (layer > 0)
                for (int input = 0; input < Max(nodesInEachLayer); input++)
                {
                    weightedSum += weights[layer][node][input] * CalculateZ(layer - 1, input);
                }
            return weightedSum + this.biases[layer][node];
        }
        private double[] CalculateZ(int layer)
        {
            double[] zs = new double[Max(nodesInEachLayer)];
            for (int i = 0; i < zs.Length; i++)
            {
                zs[i] = CalculateZ(layer, i);
            }
            return zs;
        }
        private void SetUpBiasesAndWeights()
        {
            int maxNumberOfNodesInLayer = Max(nodesInEachLayer);

            for (int layer = 0; layer < NumberOfLayers; layer++)
            {
                this.biases[layer] = new double[maxNumberOfNodesInLayer];
                this.weights[layer] = new double[maxNumberOfNodesInLayer][];

                for (int node = 0; node < maxNumberOfNodesInLayer; node++)
                {
                    weights[layer][node] = new double[maxNumberOfNodesInLayer];

                    if (node < nodesInEachLayer[layer])
                    {
                        biases[layer][node] = RandomNum(-0.5, 0.5);
                        for (int i = 0; i < maxNumberOfNodesInLayer; i++)
                            weights[layer][node][i] = RandomNum(-0.5, 0.5);
                    }
                    else
                    {
                        biases[layer][node] = 0;
                        for (int i = 0; i < maxNumberOfNodesInLayer; i++)
                            weights[layer][node][i] = 0.0;
                    }
                }
            }
            velocity = ConstantMultiply(0, this.weights);
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
        private double[] SigmoidIgnoreZero(double[] xs)
        {
            double[] sp = new double[xs.Length];
            for (int i = 0; i < sp.Length; i++)
                sp[i] = (xs[i]==0) ? 0 : Sigmoid(xs[i]);
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
        private int Max(int[] nums)
        {
            int biggest = int.MinValue;
            for (int i = 0; i < nums.Length; i++)
                if (nums[i] > biggest)
                    biggest = nums[i];
            return biggest;
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
        private double[][][] ConstantMultiply(double a, double[][][] b)
        {
            double[][][] atimesb = new double[b.Length][][];
            for (int i = 0; i < atimesb.Length; i++)
                atimesb[i] = ConstantMultiply(a, b[i]);
            return atimesb;
        }
        private double[][] Transpose(double[][] a)
        {
            //helper variables to clean up code
            int rows = a.Length;
            int columns = a[0].Length;
            int totalAmt = rows * columns;

            //our returning array (will instantiate rows soon)
            double[][] aT = new double[columns][];

            //iterating through each element in the 2D array
            for(int i = 0; i < totalAmt; i++)
            {
                //instantiate columns for the first few iterations
                if(i<columns)
                    aT[i] = new double[rows];

                //fill in each element in cycled order (thanks to modulus)
                aT[i % columns][i % rows] = a[i % rows][i % columns];
            }

            //done
            return aT;
        }
        private double[] MatrixMultiply(double[][] a, double[] v)
        {
            double[] resultant = new double[v.Length];

            for (int row = 0; row < a.Length; row++)
            {
                //linearly transform each row vector by dotting the a_row vector and v_col vector
                for (int column = 0; column < v.Length; column++)
                    resultant[row] += a[row][column] * v[column];
            }

            return resultant;
        }
        private double[] MatrixAdd(double[] a, double[] b)
        {
            double[] ab = new double[a.Length];
            for (int i = 0; i < ab.Length; i++)
                //modulus means just wrap around if we get to the end of one vector
                //this lets us "add" a 1D vector with an n-D vector for example:
                //<5> + <1,2,3,4,5> = <6,7,8,9,10>
                ab[i] = a[i%a.Length] + b[i%b.Length];
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
            int[] indexes = new int[count];

            for(int i = 0; i < count; i++)
            {
                //generate random integer
                int rand = rndm.Next(l.Count);
                
                //scan indexes for repeats
                for (int j = 0; j < i + 1; j++)
                    if (indexes[j] == rand)
                        i=j; //if so, then replace the element that was placed first

                //always assign
                indexes[i] = rand;
            }

            return indexes.ToArray();
        }
    }
    struct Errors
    {
        public double[][] partial_B;
        public double[][][] partial_W;
    }
    struct IOBatch
    {
        public List<double[]> InputList;
        public List<double[]> OutputList;
    }
}
