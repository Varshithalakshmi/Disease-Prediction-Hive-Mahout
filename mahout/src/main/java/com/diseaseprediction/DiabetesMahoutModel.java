
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

public class DiabetesMahoutModel {

    private static final int FEATURES = 21;
    private static final String DATASET = "../data/raw/diabetes.csv";

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

        Metrics(List<Record> data,
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

    private static List<Record> loadDataset() throws IOException {

        List<Record> records = new ArrayList<>();

        try (BufferedReader reader =
                     new BufferedReader(new FileReader(DATASET))) {

            String line;
            boolean header = true;

            while ((line = reader.readLine()) != null) {

                if (header) {
                    header = false;
                    continue;
                }

                String[] columns = line.split(",");

                if (columns.length < 23) {
                    continue;
                }

                int label = Integer.parseInt(columns[1]);
                double[] values = new double[FEATURES];

                for (int i = 0; i < FEATURES; i++) {
                    values[i] = Double.parseDouble(columns[i + 2]);
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

    public static void main(String[] args) throws Exception {

        System.out.println("Loading diabetes dataset...");

        List<Record> records = loadDataset();

        System.out.println("Total records: " + records.size());

        // Separate classes for stratified splitting
        List<Record> negative = new ArrayList<>();
        List<Record> positive = new ArrayList<>();

        for (Record record : records) {
            if (record.label == 1) {
                positive.add(record);
            } else {
                negative.add(record);
            }
        }

        Random random = new Random(42);

        Collections.shuffle(negative, random);
        Collections.shuffle(positive, random);

        List<Record> training = new ArrayList<>();
        List<Record> validation = new ArrayList<>();
        List<Record> testing = new ArrayList<>();

        splitClass(negative, training, validation, testing);
        splitClass(positive, training, validation, testing);

        Collections.shuffle(training, random);
        Collections.shuffle(validation, random);
        Collections.shuffle(testing, random);

        System.out.println("Training records: " + training.size());
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

        // Oversample minority class in training data only
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

        // Select threshold using validation data only
        System.out.println("\nSelecting threshold using validation data...");

        double bestThreshold = 0.50;
        double bestF1 = -1.0;

        for (double threshold = 0.10;
             threshold <= 0.90;
             threshold += 0.05) {

            Metrics validationMetrics =
                    new Metrics(validation, model, threshold);

            System.out.printf(
                    "Threshold: %.2f | Precision: %.4f | Recall: %.4f | F1: %.4f%n",
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

        // Final evaluation on untouched test data
        System.out.println("\nEvaluating final test data...");

        Metrics testMetrics =
                new Metrics(testing, model, bestThreshold);

        printMetrics("FINAL MAHOUT TEST RESULTS", testMetrics);

        // Display first ten test predictions
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

        System.out.println("\nMahout model completed.");
    }

    private static void splitClass(
            List<Record> source,
            List<Record> training,
            List<Record> validation,
            List<Record> testing) {

        int trainEnd = (int) (source.size() * 0.70);
        int validationEnd = (int) (source.size() * 0.85);

        training.addAll(source.subList(0, trainEnd));

        validation.addAll(
                source.subList(trainEnd, validationEnd)
        );

        testing.addAll(
                source.subList(validationEnd, source.size())
        );
    }
}
