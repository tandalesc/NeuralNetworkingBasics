using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkingBasics
{
    class Program
    {
        static void Main(string[] args)
        {
            Network n = new Network(3, 3);

            List<double[]> inputs = new List<double[]>();
            List<double[]> outputs = new List<double[]>();

            inputs.Add(new double[] { 0, 0, 0 }); outputs.Add(new double[] { 0, 0, 0});
            inputs.Add(new double[] { 0, 0, 1 }); outputs.Add(new double[] { 1, 0, 0});
            inputs.Add(new double[] { 0, 1, 0 }); outputs.Add(new double[] { 0, 1, 0});
            inputs.Add(new double[] { 0, 1, 1 }); outputs.Add(new double[] { 1, 1, 0});
            inputs.Add(new double[] { 1, 0, 0 }); outputs.Add(new double[] { 0, 0, 1});
            inputs.Add(new double[] { 1, 0, 1 }); outputs.Add(new double[] { 1, 0, 1});
            inputs.Add(new double[] { 1, 1, 0 }); outputs.Add(new double[] { 0, 1, 1});
            inputs.Add(new double[] { 1, 1, 1 }); outputs.Add(new double[] { 1, 1, 1});
            inputs.Add(new double[] { 0, 0,-1 }); outputs.Add(new double[] { 0, 0, 1});
            inputs.Add(new double[] { 0,-1, 0 }); outputs.Add(new double[] { 0, 1, 0});
            inputs.Add(new double[] {-1, 0, 0 }); outputs.Add(new double[] { 1, 0, 0});
            inputs.Add(new double[] {-1,-1,-1 }); outputs.Add(new double[] { 1, 1, 1});
            inputs.Add(new double[] { 0, 0, 0 }); outputs.Add(new double[] { 0, 0, 0});
            inputs.Add(new double[] { 0, 0, 1 }); outputs.Add(new double[] { 1, 0, 0});
            inputs.Add(new double[] { 0, 1, 0 }); outputs.Add(new double[] { 0, 1, 0});
            inputs.Add(new double[] { 0, 1, 1 }); outputs.Add(new double[] { 1, 1, 0});
            inputs.Add(new double[] { 1, 0, 0 }); outputs.Add(new double[] { 0, 0, 1});
            inputs.Add(new double[] { 1, 0, 1 }); outputs.Add(new double[] { 1, 0, 1});
            inputs.Add(new double[] { 1, 1, 0 }); outputs.Add(new double[] { 0, 1, 1});
            inputs.Add(new double[] { 1, 1, 1 }); outputs.Add(new double[] { 1, 1, 1});
            inputs.Add(new double[] { 0, 0,-1 }); outputs.Add(new double[] { 0, 0, 1});
            inputs.Add(new double[] { 0,-1, 0 }); outputs.Add(new double[] { 0, 1, 0});
            inputs.Add(new double[] {-1, 0, 0 }); outputs.Add(new double[] { 1, 0, 0});
            inputs.Add(new double[] {-1,-1,-1 }); outputs.Add(new double[] { 1, 1, 1});


            n.GradientDescent(inputs, outputs, 10000, 12, 0.01, 5);

            Print(n.FeedForward(-1, 0, 0));
            Print(n.FeedForward(0, 1, 0));
            Print(n.FeedForward(0, 1, 1));
            Print(n.FeedForward(0,0, 0));


            Console.Read();
        }

        static void Print(double[] vector)
        {
            Console.Write("<");
            foreach (double v in vector)
                Console.Write("{0}, ", v);
            Console.Write(">\n");
        }
    }
}
