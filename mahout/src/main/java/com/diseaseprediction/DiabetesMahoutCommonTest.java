
package com.diseaseprediction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.UniformPrior;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class DiabetesMahoutCommonTest {

    private static final int FEATURES = 21;

    private static final String TRAIN_FILE =
            "../data/processed/train.csv";

    private static final String TEST_FILE =
            "../data/processed/test.csv";

    private static final double[] SCALES = {
        1, 1, 1, 100, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 5, 30, 30, 1, 1, 13, 6, 8
    };

    static class Record {
        int label;
        double[] values;

        Record(int label, double[] values) {
            this.label = label;
            this.values = values;
        }
    }

    static class Metrics {
        int tp, tn, fp, fn;
        double accuracy, precision, recall, f1;

        Metrics(
                List<Record> data,
                OnlineLogisticRegression model,
                double threshold) {

            for (Record record : data) {

                Vector vector = createVector(record.values);

                double score = model.classifyScalar(vector);

                int prediction = score >= threshold ? 1 : 0;

                if (record.label == 1 && prediction == 1) {
                    tp++;
                } else if (record.label == 0 && prediction == 0) {
                    tn++;
                } else if (record.label == 0 && prediction == 1) {
                    fp++;
                } else {
                    fn++;
                }
            }

            int total = tp + tn + fp + fn;

            accuracy = total == 0 ? 0 :
                    (double) (tp + tn) / total;

            precision = tp + fp == 0 ? 0 :
                    (double) tp / (tp + fp);

            recall = tp + fn == 0 ? 0 :
                    (double) tp / (tp + fn);

            f1 = precision + recall == 0 ? 0 :
                    2 * precision * recall / (precision + recall);
        }
    }

    private static Vector createVector(double[] values) {

        Vector vector = new DenseVector(FEATURES + 1);

        vector.setQuick(0, 1.0);

        for (int i = 0; i < FEATURES; i++) {
            vector.setQuick(i + 1, values[i] / SCALES[i]);
        }

        return vector;
    }

    // Read the processed CSV files.
    // First 21 columns are features; last column is the target.
    private static List<Record> loadDataset(String filePath)
            throws IOException {

        List<Record> records = new ArrayList<>();

        try (BufferedReader reader =
                     new BufferedReader(new FileReader(filePath))) {

            String line;
            boolean header = true;

            while ((line = reader.readLine()) != null) {

                if (header) {
                    header = false;
                    continue;
                }

                String[] columns = line.split(",");

                if (columns.length < FEATURES + 1) {
                    continue;
                }

                int label =
                        Integer.parseInt(columns[FEATURES]);

                double[] values = new double[FEATURES];

                for (int i = 0; i < FEATURES; i++) {
                    values[i] = Double.parseDouble(columns[i]);
                }

                records.add(new Record(label, values));
            }
        }

        return records;
    }

    private static void printMetrics(String title, Metrics m) {

        System.out.println("\n===== " + title + " =====");

        System.out.printf("Accuracy : %.4f%n", m.accuracy);
        System.out.printf("Precision: %.4f%n", m.precision);
        System.out.printf("Recall   : %.4f%n", m.recall);
        System.out.printf("F1 Score : %.4f%n", m.f1);

        System.out.println("\nConfusion Matrix:");
        System.out.println("True Negative : " + m.tn);
        System.out.println("False Positive: " + m.fp);
        System.out.println("False Negative: " + m.fn);
        System.out.println("True Positive : " + m.tp);
    }

    // Split only the training dataset into training and validation.
    // The separate test dataset is never split or used for threshold selection.
    private static void splitClass(
            List<Record> source,
            List<Record> training,
            List<Record> validation) {

        int trainEnd = (int) (source.size() * 0.85);

        training.addAll(source.subList(0, trainEnd));

        validation.addAll(
                source.subList(trainEnd, source.size())
        );
    }

    public static void main(String[] args) throws Exception {

        System.out.println("Loading shared training dataset...");

        List<Record> trainingData = loadDataset(TRAIN_FILE);

        System.out.println("Loading shared testing dataset...");

        List<Record> testing = loadDataset(TEST_FILE);

        System.out.println("Training dataset records: "
                + trainingData.size());

        System.out.println("Testing dataset records: "
                + testing.size());

        // Separate training records by class.
        List<Record> negative = new ArrayList<>();
        List<Record> positive = new ArrayList<>();

        for (Record record : trainingData) {

            if (record.label == 1) {
                positive.add(record);
            } else {
                negative.add(record);
            }
        }

        Random random = new Random(42);

        Collections.shuffle(negative, random);
        Collections.shuffle(positive, random);

        // Create training and validation sets from train.csv only.
        List<Record> training = new ArrayList<>();
        List<Record> validation = new ArrayList<>();

        splitClass(negative, training, validation);
        splitClass(positive, training, validation);

        Collections.shuffle(training, random);
        Collections.shuffle(validation, random);

        System.out.println("\nTraining records: " + training.size());
        System.out.println("Validation records: " + validation.size());
        System.out.println("Testing records: " + testing.size());

        System.out.println("\nTraining Mahout model...");

        OnlineLogisticRegression model =
                new OnlineLogisticRegression(
                        2,
                        FEATURES + 1,
                        new UniformPrior()
                );

        model.learningRate(0.1);
        model.lambda(0.0001);

        // Oversample minority class in training data only.
        Random trainingRandom = new Random(42);

        for (int epoch = 1; epoch <= 5; epoch++) {

            Collections.shuffle(training, trainingRandom);

            for (Record record : training) {

                Vector vector = createVector(record.values);

                if (record.label == 1) {

                    for (int repeat = 0; repeat < 5; repeat++) {
                        model.train(record.label, vector);
                    }

                } else {
                    model.train(record.label, vector);
                }
            }

            System.out.println("Completed epoch " + epoch);
        }

        // Select the threshold using validation data only.
        System.out.println(
                "\nSelecting threshold using validation data..."
        );

        double bestThreshold = 0.50;
        double bestF1 = -1.0;

        for (int step = 10; step <= 90; step += 5) {

            double threshold = step / 100.0;

            Metrics validationMetrics =
                    new Metrics(validation, model, threshold);

            System.out.printf(
                    "Threshold: %.2f | Precision: %.4f | "
                    + "Recall: %.4f | F1: %.4f%n",
                    threshold,
                    validationMetrics.precision,
                    validationMetrics.recall,
                    validationMetrics.f1
            );

            if (validationMetrics.f1 > bestF1) {

                bestF1 = validationMetrics.f1;
                bestThreshold = threshold;
            }
        }

        System.out.printf(
                "\nSelected threshold: %.2f%n",
                bestThreshold
        );

        // Final evaluation on the untouched shared test dataset.
        System.out.println(
                "\nEvaluating shared test dataset..."
        );

        Metrics testMetrics =
                new Metrics(testing, model, bestThreshold);

        printMetrics(
                "FINAL MAHOUT COMMON TEST RESULTS",
                testMetrics
        );

        // Display the first ten test predictions.
        System.out.println("\nFirst 10 test predictions:");

        for (int i = 0; i < Math.min(10, testing.size()); i++) {

            Record record = testing.get(i);

            double score = model.classifyScalar(
                    createVector(record.values)
            );

            int prediction = score >= bestThreshold ? 1 : 0;

            System.out.printf(
                    "Actual: %d | Score: %.4f | Predicted: %d%n",
                    record.label,
                    score,
                    prediction
            );
        }

        System.out.println(
                "\nMahout common test evaluation completed."
        );
    }
}
