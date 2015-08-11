using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using VB = Microsoft.VisualBasic;

namespace NeuralNetworkingBasics
{
    class FileIO
    {
        private static string[] ReadFile(string fileName)
        {
            List<string> outputStream = new List<string>();
            try
            {
                using (VB.FileIO.TextFieldParser parser = new VB.FileIO.TextFieldParser(@fileName))
                {
                    while (!parser.EndOfData)
                    {
                        outputStream.Add(parser.ReadLine());
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("The file could not be read");
            }
            return outputStream.ToArray();
        }
        private static bool WriteFile(string fileName, string[] fileContents)
        {
            try
            {
                using (StreamWriter sw = new StreamWriter(fileName))
                {
                    int lastLine = fileContents.Length;
                    for (int line = 0; line < lastLine; line++)
                    {
                        sw.WriteLine(fileContents[line]);
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("The file could not be written");
                return false;
            }
            //only if exception is not thrown
            return true;
        }
        private static double[] BuildVector(string vector)
        {
            vector.Trim();
            string[] data;

            if(vector.Length > 1)
                 data = vector.Split('|');
            else
                 data = new string[] {vector};

            List<double> vData = new List<double>();
            for (int component = 0; component < data.Length; component++)
            {
                double d;
                double.TryParse(data[component], out d);

                vData.Add(d);
            }

            return vData.ToArray();
        }

        public static FileData UnwrapCSVFile(string fileName)
        {
            string[] fileData = ReadFile(fileName);


            List<double[]> inputs = new List<double[]>();
            List<double[]> outputs = new List<double[]>();
            List<double[]> inputSet = new List<double[]>();

            for (int line = 0; line < fileData.Length; line++)
            {
                string[] iopair_string = fileData[line].Split(',');
                if(iopair_string.Length == 2 && !iopair_string[1].Equals(string.Empty)) //it's a legit input output pair, not a sample
                {
                    inputs.Add(BuildVector(iopair_string[0]));
                    outputs.Add(BuildVector(iopair_string[1]));
                }
                else 
                {
                    inputSet.Add(BuildVector(iopair_string[0]));
                }
            }

            IOBatch i = new IOBatch();
            i.InputList = inputs;
            i.OutputList = outputs;

            FileData returnData = new FileData();
            returnData.LearningData = i;
            returnData.InputSet = inputSet;

            return returnData;
        }
    }
    struct FileData
    {
        public IOBatch LearningData;
        public List<double[]> InputSet;
    }
}
