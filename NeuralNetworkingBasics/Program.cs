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
            //create and design network
            Network n = new Network(3, 1);

            //organize input output pairs
            List<double[]> inputs = new List<double[]>();
            List<double[]> outputs = new List<double[]>();

            inputs.Add(new double[] { 0, 0, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0, 0, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0, 0, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0, 0, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0, 0, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0, 0, 1 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 0, 1, 0 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 0, 1, 1 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 1, 0, 0 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 1, 0, 1 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 1, 1, 0 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 1, 1, 1 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 0, 0,-1 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0,-1, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] {-1, 0, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] {-1,-1,-1 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0, 0, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0, 0, 1 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 0, 1, 0 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 0, 1, 1 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 1, 0, 0 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 1, 0, 1 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 1, 1, 0 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 1, 1, 1 }); outputs.Add(new double[] { 1});
            inputs.Add(new double[] { 0, 0,-1 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] { 0,-1, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] {-1, 0, 0 }); outputs.Add(new double[] { 0});
            inputs.Add(new double[] {-1,-1,-1 }); outputs.Add(new double[] { 0});

            //tell network to learn from inputs
            n.GradientDescent(inputs, outputs, 500, 8, 0.15, 80);

            //run the following inputs and output the results
            Print(n.Run(-1, 0, 0));
            Print(n.Run(1, 0, 0));
            Print(n.Run(0, 1, 0));
            Print(n.Run(0, 1, 1));
            Print(n.Run(0, 0, -1));
            Print(n.Run(0, 0, 0));


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
