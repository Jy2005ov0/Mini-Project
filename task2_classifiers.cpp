// ============================================================================
// UCCC2513 Mini Project — Task 2: Traffic Sign Recognition
//
// Use the best feature from Task 1 with 3 machine learning classifiers:
//   1. KNN
//   2. ANN
//   3. Naive Bayes
//
// Evaluation:
//   - Accuracy
//   - Precision
//   - Recall
//   - F1-score
//   - Confusion matrix
//   - Visualization of results
//
// Selected Task 1 feature: HOG (144 dimensions)
// ============================================================================

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::ml;

// Shared with main()'s single-CSV evaluation and the external-training
// evaluation added below, so both always agree on the feature width.
static const int HOG_FEATURES = 144;

// NormalBayesClassifier fits a full covariance matrix per class, which is
// singular whenever a class has fewer training examples than feature
// dimensions - with up to 57 classes and some having as few as 2 external
// training images, that's true for most classes at 144-dim. A singular
// covariance makes that class's discriminant score blow up and win every
// single prediction regardless of input (observed: Naive Bayes predicted
// the same one class for all 84 test images). Reducing to this many PCA
// dimensions (fit on training data only) keeps the estimate stable for any
// class with >10 real examples; classes below that remain fundamentally
// hard for any classifier - same small-data limit as elsewhere in this
// project, not something dimensionality reduction can fix.
static const int NAIVE_BAYES_PCA_DIMS = 10;

// ============================================================================
// Helper 1: split one CSV line
// The Task 1 CSV does not contain commas inside the fields, so a simple parser
// is enough here.
// ============================================================================

vector<string> splitCSV(const string& line) {
    vector<string> cells;
    string cell;
    stringstream ss(line);

    while (getline(ss, cell, ','))
        cells.push_back(cell);

    // Keep the final empty field, if any
    if (!line.empty() && line.back() == ',')
        cells.push_back("");

    return cells;
}

// ============================================================================
// Helper 2: structure for one row from Task 1
// We only keep Filename, SignID and the 144 HOG values for Task 2.
// ============================================================================

struct Sample {
    string filename;
    string signID;
    vector<float> hog;
};

// ============================================================================
// Helper 3: read Task 1 CSV
// ============================================================================

// Original 144-column selection (hog0..hog143), kept as the default so the
// existing single-CSV evaluation in main() is untouched.
vector<string> defaultHogColumns() {
    vector<string> cols;
    for (int h = 0; h < HOG_FEATURES; h++) cols.push_back("hog" + to_string(h));
    return cols;
}

// featureColumns lets a caller load an arbitrary named set of columns
// instead of the hardcoded hog0..hog143 - used by evaluateWithExternalTraining()
// below to load the exact same ANOVA-selected feature set the Python side
// uses, for a fair classifier-vs-classifier comparison (same information,
// different algorithm) rather than comparing different feature engineering.
vector<Sample> loadTask1CSV(const string& csvPath, const vector<string>& featureColumns = defaultHogColumns()) {
    vector<Sample> samples;
    ifstream in(csvPath);

    if (!in.is_open()) {
        cout << "Error: could not open " << csvPath << endl;
        return samples;
    }

    string headerLine;
    getline(in, headerLine);
    vector<string> header = splitCSV(headerLine);

    int filenameCol = -1;
    int signIDCol = -1;
    int numFeatures = (int)featureColumns.size();
    vector<int> featureCols(numFeatures, -1);

    for (int c = 0; c < (int)header.size(); c++) {
        if (header[c] == "Filename") filenameCol = c;
        if (header[c] == "SignID") signIDCol = c;

        for (int h = 0; h < numFeatures; h++) {
            if (header[c] == featureColumns[h])
                featureCols[h] = c;
        }
    }

    if (filenameCol < 0 || signIDCol < 0) {
        cout << "Error: Filename or SignID column is missing." << endl;
        return samples;
    }

    for (int h = 0; h < numFeatures; h++) {
        if (featureCols[h] < 0) {
            cout << "Error: " << featureColumns[h] << " column is missing." << endl;
            return samples;
        }
    }

    string line;
    while (getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        vector<string> cells = splitCSV(line);
        if ((int)cells.size() < (int)header.size()) continue;

        Sample s;
        s.filename = cells[filenameCol];
        s.signID = cells[signIDCol];
        s.hog.resize(numFeatures, 0.0f);

        bool valid = true;
        for (int h = 0; h < numFeatures; h++) {
            try {
                s.hog[h] = stof(cells[featureCols[h]]);
            }
            catch (...) {
                valid = false;
                break;
            }
        }

        if (valid)
            samples.push_back(s);
    }

    return samples;
}

// ============================================================================
// Helper 4: normalization
// Same idea as the lecturer's examples:
// Xnew = (X - mean) / sigma
//
// IMPORTANT: mean and sigma are calculated from TRAINING DATA only.
// The same values are then used to normalize verification data.
// ============================================================================

void normalizeTrainingData(Mat& trainingData,
    Mat& verificationData,
    vector<double>& mean,
    vector<double>& sigma) {

    mean.clear();
    sigma.clear();

    for (int i = 0; i < trainingData.cols; i++) {
        Scalar meanOut, sigmaOut;
        meanStdDev(trainingData.col(i), meanOut, sigmaOut);

        double m = meanOut[0];
        double s = sigmaOut[0];

        // Avoid division by zero if one HOG dimension is constant
        if (s < 1e-9) s = 1.0;

        mean.push_back(m);
        sigma.push_back(s);
    }

    for (int i = 0; i < trainingData.cols; i++) {
        trainingData.col(i) = (trainingData.col(i) - mean[i]) / sigma[i];
        verificationData.col(i) = (verificationData.col(i) - mean[i]) / sigma[i];
    }
}

// ============================================================================
// Evaluation result
// ============================================================================

struct EvaluationResult {
    string modelName;
    double accuracy = 0.0;
    double precision = 0.0;  // macro precision
    double recall = 0.0;     // macro recall
    double f1 = 0.0;         // macro F1-score
    Mat confusion;
    vector<double> classPrecision;
    vector<double> classRecall;
    vector<double> classF1;
};

// ============================================================================
// Helper 5: calculate accuracy, precision, recall, F1 and confusion matrix
// ============================================================================

EvaluationResult evaluateModel(const string& modelName,
    const vector<int>& truth,
    const vector<int>& predicted,
    int numberOfClasses) {

    EvaluationResult result;
    result.modelName = modelName;
    result.confusion = Mat::zeros(numberOfClasses, numberOfClasses, CV_32S);

    int correct = 0;

    for (int i = 0; i < (int)truth.size(); i++) {
        int actual = truth[i];
        int pred = predicted[i];

        if (actual >= 0 && actual < numberOfClasses &&
            pred >= 0 && pred < numberOfClasses) {
            result.confusion.at<int>(actual, pred)++;
        }

        if (actual == pred) correct++;
    }

    result.accuracy = truth.empty() ? 0.0 : (double)correct / truth.size();

    result.classPrecision.resize(numberOfClasses, 0.0);
    result.classRecall.resize(numberOfClasses, 0.0);
    result.classF1.resize(numberOfClasses, 0.0);

    double precisionSum = 0.0;
    double recallSum = 0.0;
    double f1Sum = 0.0;

    for (int c = 0; c < numberOfClasses; c++) {
        int TP = result.confusion.at<int>(c, c);
        int FP = 0;
        int FN = 0;

        for (int r = 0; r < numberOfClasses; r++) {
            if (r != c) FP += result.confusion.at<int>(r, c);
        }

        for (int k = 0; k < numberOfClasses; k++) {
            if (k != c) FN += result.confusion.at<int>(c, k);
        }

        double p = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0;
        double r = (TP + FN > 0) ? (double)TP / (TP + FN) : 0.0;
        double f = (p + r > 0.0) ? 2.0 * p * r / (p + r) : 0.0;

        result.classPrecision[c] = p;
        result.classRecall[c] = r;
        result.classF1[c] = f;

        precisionSum += p;
        recallSum += r;
        f1Sum += f;
    }

    result.precision = precisionSum / numberOfClasses;
    result.recall = recallSum / numberOfClasses;
    result.f1 = f1Sum / numberOfClasses;

    return result;
}

// ============================================================================
// Helper 6: show one confusion matrix as an OpenCV image
// Rows = actual class, columns = predicted class
// ============================================================================

Mat drawConfusionMatrix(const EvaluationResult& result,
    const vector<string>& classNames) {

    int n = (int)classNames.size();
    int cell = 40;
    int leftMargin = 95;
    int topMargin = 95;
    int bottomMargin = 55;

    int width = leftMargin + n * cell + 20;
    int height = topMargin + n * cell + bottomMargin;

    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));

    putText(image, result.modelName + " Confusion Matrix",
        Point(15, 28), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);

    putText(image, "Predicted class",
        Point(leftMargin, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

    putText(image, "Actual",
        Point(10, topMargin - 15), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);

    int maxValue = 1;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            maxValue = max(maxValue, result.confusion.at<int>(r, c));

    for (int c = 0; c < n; c++) {
        putText(image, classNames[c],
            Point(leftMargin + c * cell + 4, topMargin - 12),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 0), 1);
    }

    for (int r = 0; r < n; r++) {
        putText(image, classNames[r],
            Point(48, topMargin + r * cell + 25),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 0), 1);

        for (int c = 0; c < n; c++) {
            int value = result.confusion.at<int>(r, c);
            int intensity = (int)(220.0 * value / maxValue);

            // Stronger count = darker cell
            Scalar fillColour(255 - intensity, 255 - intensity, 255);
            Rect box(leftMargin + c * cell, topMargin + r * cell, cell, cell);
            rectangle(image, box, fillColour, FILLED);
            rectangle(image, box, Scalar(180, 180, 180), 1);

            Scalar textColour = (intensity > 130) ? Scalar(255, 255, 255) : Scalar(0, 0, 0);
            putText(image, to_string(value),
                Point(box.x + 14, box.y + 25),
                FONT_HERSHEY_SIMPLEX, 0.4, textColour, 1);
        }
    }

    return image;
}

// ============================================================================
// Helper 7: visualization comparing the 3 classifiers
// ============================================================================

Mat drawModelComparison(const vector<EvaluationResult>& results) {
    const int width = 950;
    const int height = 600;
    const int left = 90;
    const int right = 35;
    const int top = 70;
    const int bottom = 100;

    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));

    putText(image, "Task 2 - Classifier Performance",
        Point(220, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);

    int plotW = width - left - right;
    int plotH = height - top - bottom;

    // Axis
    line(image, Point(left, top), Point(left, top + plotH), Scalar(0, 0, 0), 2);
    line(image, Point(left, top + plotH), Point(left + plotW, top + plotH), Scalar(0, 0, 0), 2);

    // 0.0 to 1.0 scale
    for (int i = 0; i <= 5; i++) {
        double value = i * 0.2;
        int y = top + plotH - (int)(value * plotH);

        line(image, Point(left - 5, y), Point(left + plotW, y), Scalar(220, 220, 220), 1);

        stringstream ss;
        ss << fixed << setprecision(1) << value;
        putText(image, ss.str(), Point(45, y + 5),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
    }

    vector<string> metricNames = { "Accuracy", "Precision", "Recall", "F1-score" };
    vector<Scalar> modelColours = {
        Scalar(220, 120, 40),
        Scalar(80, 170, 80),
        Scalar(60, 80, 220)
    };

    int groups = 4;
    int groupWidth = plotW / groups;
    int barWidth = 34;
    int gap = 8;

    for (int m = 0; m < groups; m++) {
        int groupX = left + m * groupWidth;

        for (int model = 0; model < (int)results.size(); model++) {
            double value = 0.0;
            if (m == 0) value = results[model].accuracy;
            if (m == 1) value = results[model].precision;
            if (m == 2) value = results[model].recall;
            if (m == 3) value = results[model].f1;

            int x = groupX + 35 + model * (barWidth + gap);
            int barHeight = (int)(value * plotH);
            int y = top + plotH - barHeight;

            rectangle(image, Rect(x, y, barWidth, barHeight),
                modelColours[model], FILLED);

            stringstream ss;
            ss << fixed << setprecision(2) << value;
            putText(image, ss.str(), Point(x - 2, y - 6),
                FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 0), 1);
        }

        putText(image, metricNames[m],
            Point(groupX + 35, top + plotH + 35),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    // Legend
    int legendY = height - 35;
    for (int i = 0; i < (int)results.size(); i++) {
        int x = 260 + i * 180;
        rectangle(image, Rect(x, legendY - 15, 20, 20), modelColours[i], FILLED);
        putText(image, results[i].modelName, Point(x + 28, legendY),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    return image;
}

// Reads one column name per line - used to load the exact feature set the
// Python side's SelectKBest chose (shared_selected_features.txt), so this
// evaluation compares classifiers on identical information rather than
// different feature engineering.
vector<string> readColumnList(const string& path) {
    vector<string> cols;
    ifstream in(path);
    if (!in.is_open()) return cols;
    string line;
    while (getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) cols.push_back(line);
    }
    return cols;
}

// ============================================================================
// Helper 8: external-training evaluation
//
// Train KNN/ANN/Naive Bayes on TSRD_external only, then predict on the
// original 84 project images - the same protocol as the Python side's
// Stage 0, so the two halves of Task 2 can be compared head-to-head on the
// exact same held-out test set. Reuses normalizeTrainingData() and
// evaluateModel() rather than reimplementing them, so this can't silently
// drift from the single-CSV evaluation's own logic. Also loads the same
// feature COLUMNS the Python side selected (shared_selected_features.txt),
// instead of this file's own 144-column HOG truncation, so this is a fair
// classifier-vs-classifier comparison, not a features-vs-features one.
// ============================================================================

void evaluateWithExternalTraining(const string& trainCsvPath, const string& testCsvPath) {
    if (!std::filesystem::exists(trainCsvPath)) {
        cout << "\nExternal-training evaluation skipped: " << trainCsvPath << " not found.\n";
        return;
    }

    vector<string> featureColumns = readColumnList("shared_selected_features.txt");
    if (featureColumns.empty()) {
        cout << "\nshared_selected_features.txt not found or empty - falling back to "
             << "this file's own " << HOG_FEATURES << "-column HOG selection "
             << "(run task2_combined.py's Stage 0 first to generate a fair shared "
             << "feature list).\n";
        featureColumns = defaultHogColumns();
    }
    int numFeatures = (int)featureColumns.size();
    cout << "\nUsing " << numFeatures << " shared feature column(s) for this evaluation.\n";

    vector<Sample> trainSamples = loadTask1CSV(trainCsvPath, featureColumns);
    vector<Sample> testSamples = loadTask1CSV(testCsvPath, featureColumns);
    if (trainSamples.empty() || testSamples.empty()) {
        cout << "\nExternal-training evaluation skipped: could not load data.\n";
        return;
    }

    cout << "\n================================================================\n";
    cout << "EXTERNAL-TRAINING EVALUATION (train on TSRD_external, test on our 84)\n";
    cout << "================================================================\n";
    cout << "Training images (external): " << trainSamples.size() << "\n";
    cout << "Test images (ours)        : " << testSamples.size() << "\n";

    // Class universe = every SignID seen in either set, so a class that only
    // appears in the test set (zero external examples) still gets a slot -
    // it will simply always be misclassified, same as the Python side.
    set<string> classSet, trainClasses;
    for (auto& s : trainSamples) { classSet.insert(s.signID); trainClasses.insert(s.signID); }
    for (auto& s : testSamples) classSet.insert(s.signID);
    vector<string> classNames(classSet.begin(), classSet.end());
    int numberOfClasses = (int)classNames.size();
    map<string, int> classToNumber;
    for (int i = 0; i < numberOfClasses; i++) classToNumber[classNames[i]] = i;

    int zeroExampleClasses = 0;
    for (auto& c : classNames) if (!trainClasses.count(c)) zeroExampleClasses++;
    if (zeroExampleClasses)
        cout << zeroExampleClasses << " class(es) have zero external training examples "
             << "(always misclassified - same caveat as the Python side).\n";

    Mat trainingData((int)trainSamples.size(), numFeatures, CV_32F);
    Mat trainingLabels((int)trainSamples.size(), 1, CV_32S);
    for (int r = 0; r < (int)trainSamples.size(); r++) {
        trainingLabels.at<int>(r, 0) = classToNumber[trainSamples[r].signID];
        for (int c = 0; c < numFeatures; c++)
            trainingData.at<float>(r, c) = trainSamples[r].hog[c];
    }

    Mat verificationData((int)testSamples.size(), numFeatures, CV_32F);
    Mat verificationLabels((int)testSamples.size(), 1, CV_32S);
    for (int r = 0; r < (int)testSamples.size(); r++) {
        verificationLabels.at<int>(r, 0) = classToNumber[testSamples[r].signID];
        for (int c = 0; c < numFeatures; c++)
            verificationData.at<float>(r, c) = testSamples[r].hog[c];
    }

    vector<double> mean, sigma;
    normalizeTrainingData(trainingData, verificationData, mean, sigma);

    // ----- Train (identical setup to the single-CSV evaluation above) -----
    Ptr<KNearest> knn = KNearest::create();
    knn->setIsClassifier(true);
    knn->setDefaultK(1);
    knn->train(trainingData, ROW_SAMPLE, trainingLabels);

    Mat annTrainingLabels = Mat::zeros(trainingData.rows, numberOfClasses, CV_32F);
    for (int r = 0; r < trainingLabels.rows; r++)
        annTrainingLabels.at<float>(r, trainingLabels.at<int>(r, 0)) = 1.0f;

    Ptr<ANN_MLP> ann = ANN_MLP::create();
    Mat layerSizes = (Mat_<int>(1, 3) << numFeatures, 64, numberOfClasses);
    ann->setLayerSizes(layerSizes);
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 2000, 1e-6));
    theRNG().state = 12345;
    ann->train(trainingData, ROW_SAMPLE, annTrainingLabels);

    // See NAIVE_BAYES_PCA_DIMS above - fit PCA on training data only (no
    // leakage), then project both train and test through it just for the
    // Naive Bayes path. KNN/ANN don't estimate a covariance matrix, so they
    // aren't affected by this and keep using the full 144-dim features.
    PCA bayesPCA(trainingData, Mat(), PCA::DATA_AS_ROW, NAIVE_BAYES_PCA_DIMS);
    Mat trainingDataBayes = bayesPCA.project(trainingData);
    Mat verificationDataBayes = bayesPCA.project(verificationData);

    Ptr<NormalBayesClassifier> bayes = NormalBayesClassifier::create();
    bayes->train(trainingDataBayes, ROW_SAMPLE, trainingLabels);

    // ----- Predict on the 84 (genuinely held out - never trained on) -----
    vector<int> truth, predKNN, predANN, predBayes;
    for (int i = 0; i < verificationData.rows; i++) {
        truth.push_back(verificationLabels.at<int>(i, 0));

        Mat knnResult;
        float knnResponse = knn->findNearest(verificationData.row(i), 1, knnResult);
        predKNN.push_back(cvRound(knnResponse));

        Mat annOutput;
        ann->predict(verificationData.row(i), annOutput);
        Point maxLoc;
        minMaxLoc(annOutput, nullptr, nullptr, nullptr, &maxLoc);
        predANN.push_back(maxLoc.x);

        float bayesResponse = bayes->predict(verificationDataBayes.row(i));
        predBayes.push_back(cvRound(bayesResponse));
    }

    EvaluationResult knnResult = evaluateModel("KNN", truth, predKNN, numberOfClasses);
    EvaluationResult annResult = evaluateModel("ANN", truth, predANN, numberOfClasses);
    EvaluationResult bayesResult = evaluateModel("Naive Bayes", truth, predBayes, numberOfClasses);
    vector<EvaluationResult> allResults = { knnResult, annResult, bayesResult };

    cout << fixed << setprecision(4);
    cout << "\nClassifier     Accuracy    Precision   Recall      F1-score    \n";
    cout << string(65, '-') << "\n";
    for (const auto& r : allResults) {
        cout << left << setw(15) << r.modelName << setw(12) << r.accuracy
             << setw(12) << r.precision << setw(12) << r.recall << setw(12) << r.f1 << "\n";
    }

    // Distinct filenames so this never collides with the single-CSV mode's
    // own task2_*.csv/png output, or with the Python side's task2_external_*.
    ofstream metricsFile("task2_cpp_external_metrics.csv");
    if (metricsFile.is_open()) {
        metricsFile << "Classifier,Accuracy,Precision,Recall,F1Score\n";
        for (const auto& r : allResults)
            metricsFile << r.modelName << "," << r.accuracy << "," << r.precision << ","
                        << r.recall << "," << r.f1 << "\n";
    }

    ofstream predictionFile("task2_cpp_external_predictions.csv");
    if (predictionFile.is_open()) {
        predictionFile << "Filename,Truth,KNN,ANN,NaiveBayes\n";
        for (int i = 0; i < (int)truth.size(); i++) {
            predictionFile << testSamples[i].filename << "," << classNames[truth[i]] << ","
                           << classNames[predKNN[i]] << "," << classNames[predANN[i]] << ","
                           << classNames[predBayes[i]] << "\n";
        }
    }

    imwrite("task2_cpp_external_knn_confusion_matrix.png", drawConfusionMatrix(knnResult, classNames));
    imwrite("task2_cpp_external_ann_confusion_matrix.png", drawConfusionMatrix(annResult, classNames));
    imwrite("task2_cpp_external_bayes_confusion_matrix.png", drawConfusionMatrix(bayesResult, classNames));
    imwrite("task2_cpp_external_classifier_comparison.png", drawModelComparison(allResults));

    cout << "\nFiles written:\n"
         << "  task2_cpp_external_metrics.csv\n"
         << "  task2_cpp_external_predictions.csv\n"
         << "  task2_cpp_external_knn_confusion_matrix.png\n"
         << "  task2_cpp_external_ann_confusion_matrix.png\n"
         << "  task2_cpp_external_bayes_confusion_matrix.png\n"
         << "  task2_cpp_external_classifier_comparison.png\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    const string csvPath = "task1_combined_features.csv";

    // We hold out one image per class for verification.
    // A class therefore needs at least 3 images so that Naive Bayes still has
    // at least 2 training examples for estimating its statistics.
    const int MIN_SAMPLES_PER_CLASS = 3;

    // ========================================================================
    // Section 1: Read the Task 1 feature data
    // ========================================================================

    vector<Sample> samples = loadTask1CSV(csvPath);

    if (samples.empty()) {
        cout << "No data loaded from " << csvPath << endl;
        return -1;
    }

    cout << "Task 1 CSV loaded successfully." << endl;
    cout << "Total images in CSV: " << samples.size() << endl;

    // ========================================================================
    // Section 2: Select the best Task 1 feature
    //
    // HOG is used because SignID recognition depends strongly on the inner
    // symbol / digit / arrow of the traffic sign. Colour and outer shape are
    // useful for segmentation, but many different SignIDs share them.
    // ========================================================================

    cout << "\nSelected feature for Task 2: HOG" << endl;
    cout << "Number of HOG features       : " << HOG_FEATURES << endl;

    // Group the images according to SignID
    map<string, vector<int>> groupedRows;
    for (int i = 0; i < (int)samples.size(); i++)
        groupedRows[samples[i].signID].push_back(i);

    cout << "Total SignID classes         : " << groupedRows.size() << endl;

    // ========================================================================
    // Section 3: Keep classes that can be evaluated fairly
    // ========================================================================

    vector<string> classNames;
    vector<int> excludedCounts;

    cout << "\nClasses used for quantitative evaluation:" << endl;
    for (const auto& item : groupedRows) {
        if ((int)item.second.size() >= MIN_SAMPLES_PER_CLASS) {
            classNames.push_back(item.first);
            cout << "  SignID " << item.first << " : " << item.second.size() << " images" << endl;
        }
    }

    cout << "\nClasses not used in the train/verification experiment" << endl;
    cout << "(fewer than " << MIN_SAMPLES_PER_CLASS << " images):" << endl;
    for (const auto& item : groupedRows) {
        if ((int)item.second.size() < MIN_SAMPLES_PER_CLASS) {
            cout << "  SignID " << item.first << " : " << item.second.size() << " image(s)" << endl;
        }
    }

    if (classNames.size() < 2) {
        cout << "Not enough classes for classification." << endl;
        return -1;
    }

    int numberOfClasses = (int)classNames.size();

    // Map the three-digit SignID to 0,1,2,... for OpenCV ML
    map<string, int> classToNumber;
    for (int i = 0; i < numberOfClasses; i++)
        classToNumber[classNames[i]] = i;

    // ========================================================================
    // Section 4: Set up training data and verification data
    //
    // One image from every usable SignID is reserved for verification.
    // The remaining images of that SignID are used for training.
    // A fixed random seed is used so that every classifier sees exactly the
    // same train/verification split every time the program is run.
    // ========================================================================

    vector<int> trainingRows;
    vector<int> verificationRows;

    mt19937 rng(42);

    for (const string& signID : classNames) {
        vector<int> rows = groupedRows[signID];
        shuffle(rows.begin(), rows.end(), rng);

        verificationRows.push_back(rows[0]);
        for (int i = 1; i < (int)rows.size(); i++)
            trainingRows.push_back(rows[i]);
    }

    Mat trainingData((int)trainingRows.size(), HOG_FEATURES, CV_32F);
    Mat trainingLabels((int)trainingRows.size(), 1, CV_32S);

    Mat verificationData((int)verificationRows.size(), HOG_FEATURES, CV_32F);
    Mat verificationLabels((int)verificationRows.size(), 1, CV_32S);

    for (int r = 0; r < (int)trainingRows.size(); r++) {
        const Sample& s = samples[trainingRows[r]];
        trainingLabels.at<int>(r, 0) = classToNumber[s.signID];

        for (int c = 0; c < HOG_FEATURES; c++)
            trainingData.at<float>(r, c) = s.hog[c];
    }

    for (int r = 0; r < (int)verificationRows.size(); r++) {
        const Sample& s = samples[verificationRows[r]];
        verificationLabels.at<int>(r, 0) = classToNumber[s.signID];

        for (int c = 0; c < HOG_FEATURES; c++)
            verificationData.at<float>(r, c) = s.hog[c];
    }

    cout << "\nTraining images              : " << trainingData.rows << endl;
    cout << "Verification images          : " << verificationData.rows << endl;
    cout << "Classes evaluated            : " << numberOfClasses << endl;

    // ========================================================================
    // Section 5: Do normalization of the data
    // Xnew = (X - mean) / sigma
    // ========================================================================

    vector<double> mean, sigma;
    normalizeTrainingData(trainingData, verificationData, mean, sigma);

    // ========================================================================
    // Section 6: Train KNN
    //
    // K = 1 is used because this dataset has only a small number of training
    // images for each SignID after one image is held out for verification.
    // ========================================================================

    Ptr<KNearest> knn = KNearest::create();
    knn->setIsClassifier(true);
    knn->setDefaultK(1);
    knn->train(trainingData, ROW_SAMPLE, trainingLabels);

    // ========================================================================
    // Section 7: Train ANN
    // ========================================================================

    Mat annTrainingLabels = Mat::zeros(trainingData.rows, numberOfClasses, CV_32F);
    for (int r = 0; r < trainingLabels.rows; r++) {
        int classNo = trainingLabels.at<int>(r, 0);
        annTrainingLabels.at<float>(r, classNo) = 1.0f;
    }

    Ptr<ANN_MLP> ann = ANN_MLP::create();

    Mat layerSizes = (Mat_<int>(1, 3) << HOG_FEATURES, 64, numberOfClasses);
    ann->setLayerSizes(layerSizes);
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS,
        2000, 1e-6));

    // Fix OpenCV random state to make the ANN experiment repeatable
    theRNG().state = 12345;
    ann->train(trainingData, ROW_SAMPLE, annTrainingLabels);

    // ========================================================================
    // Section 8: Train Naive Bayes
    // ========================================================================

    Ptr<NormalBayesClassifier> bayes = NormalBayesClassifier::create();
    bayes->train(trainingData, ROW_SAMPLE, trainingLabels);

    // ========================================================================
    // Section 9: Test all 3 classifiers using the SAME verification images
    // ========================================================================

    vector<int> truth;
    vector<int> predKNN;
    vector<int> predANN;
    vector<int> predBayes;

    cout << "\nVerification results" << endl;
    cout << left
        << setw(26) << "Filename"
        << setw(8) << "Truth"
        << setw(8) << "KNN"
        << setw(8) << "ANN"
        << setw(8) << "Bayes"
        << endl;
    cout << string(58, '-') << endl;

    for (int i = 0; i < verificationData.rows; i++) {
        int actual = verificationLabels.at<int>(i, 0);
        truth.push_back(actual);

        // ----- KNN prediction -----
        Mat knnResult;
        float knnResponse = knn->findNearest(verificationData.row(i), 1, knnResult);
        int knnClass = cvRound(knnResponse);
        predKNN.push_back(knnClass);

        // ----- ANN prediction -----
        Mat annOutput;
        ann->predict(verificationData.row(i), annOutput);

        Point maxLoc;
        minMaxLoc(annOutput, nullptr, nullptr, nullptr, &maxLoc);
        int annClass = maxLoc.x;
        predANN.push_back(annClass);

        // ----- Naive Bayes prediction -----
        float bayesResponse = bayes->predict(verificationData.row(i));
        int bayesClass = cvRound(bayesResponse);
        predBayes.push_back(bayesClass);

        const Sample& sample = samples[verificationRows[i]];

        cout << left
            << setw(26) << sample.filename
            << setw(8) << classNames[actual]
            << setw(8) << classNames[knnClass]
            << setw(8) << classNames[annClass]
            << setw(8) << classNames[bayesClass]
            << endl;
    }

    // ========================================================================
    // Section 10: Accuracy, precision, recall, F1-score and confusion matrix
    // ========================================================================

    EvaluationResult knnResult = evaluateModel("KNN", truth, predKNN, numberOfClasses);
    EvaluationResult annResult = evaluateModel("ANN", truth, predANN, numberOfClasses);
    EvaluationResult bayesResult = evaluateModel("Naive Bayes", truth, predBayes, numberOfClasses);

    vector<EvaluationResult> allResults = { knnResult, annResult, bayesResult };

    cout << fixed << setprecision(4);
    cout << "\n================ Task 2 Performance =================" << endl;
    cout << left
        << setw(15) << "Classifier"
        << setw(12) << "Accuracy"
        << setw(12) << "Precision"
        << setw(12) << "Recall"
        << setw(12) << "F1-score"
        << endl;
    cout << string(63, '-') << endl;

    for (const auto& r : allResults) {
        cout << left
            << setw(15) << r.modelName
            << setw(12) << r.accuracy
            << setw(12) << r.precision
            << setw(12) << r.recall
            << setw(12) << r.f1
            << endl;
    }
    cout << "======================================================" << endl;

    // Show per-class metrics too
    for (const auto& r : allResults) {
        cout << "\n" << r.modelName << " - per-class results" << endl;
        cout << left
            << setw(10) << "SignID"
            << setw(12) << "Precision"
            << setw(12) << "Recall"
            << setw(12) << "F1-score"
            << endl;

        for (int c = 0; c < numberOfClasses; c++) {
            cout << left
                << setw(10) << classNames[c]
                << setw(12) << r.classPrecision[c]
                << setw(12) << r.classRecall[c]
                << setw(12) << r.classF1[c]
                << endl;
        }
    }

    // ========================================================================
    // Section 11: Save numerical results to CSV
    // ========================================================================

    ofstream metricsFile("task2_classifier_metrics.csv");
    if (metricsFile.is_open()) {
        metricsFile << "Classifier,Accuracy,Precision,Recall,F1Score\n";
        for (const auto& r : allResults) {
            metricsFile << r.modelName << ","
                << r.accuracy << ","
                << r.precision << ","
                << r.recall << ","
                << r.f1 << "\n";
        }
        metricsFile.close();
    }

    ofstream predictionFile("task2_predictions.csv");
    if (predictionFile.is_open()) {
        predictionFile << "Filename,Truth,KNN,ANN,NaiveBayes\n";
        for (int i = 0; i < (int)truth.size(); i++) {
            const Sample& sample = samples[verificationRows[i]];
            predictionFile << sample.filename << ","
                << classNames[truth[i]] << ","
                << classNames[predKNN[i]] << ","
                << classNames[predANN[i]] << ","
                << classNames[predBayes[i]] << "\n";
        }
        predictionFile.close();
    }

    // ========================================================================
    // Section 12: Visualization of results
    // ========================================================================

    Mat knnCM = drawConfusionMatrix(knnResult, classNames);
    Mat annCM = drawConfusionMatrix(annResult, classNames);
    Mat bayesCM = drawConfusionMatrix(bayesResult, classNames);
    Mat comparison = drawModelComparison(allResults);

    imwrite("task2_knn_confusion_matrix.png", knnCM);
    imwrite("task2_ann_confusion_matrix.png", annCM);
    imwrite("task2_bayes_confusion_matrix.png", bayesCM);
    imwrite("task2_classifier_comparison.png", comparison);

    imshow("KNN Confusion Matrix", knnCM);
    imshow("ANN Confusion Matrix", annCM);
    imshow("Naive Bayes Confusion Matrix", bayesCM);
    imshow("Classifier Comparison", comparison);

    cout << "\nFiles written:" << endl;
    cout << "  task2_classifier_metrics.csv" << endl;
    cout << "  task2_predictions.csv" << endl;
    cout << "  task2_knn_confusion_matrix.png" << endl;
    cout << "  task2_ann_confusion_matrix.png" << endl;
    cout << "  task2_bayes_confusion_matrix.png" << endl;
    cout << "  task2_classifier_comparison.png" << endl;

    // Second evaluation: same 3 classifiers, trained on TSRD_external instead
    // of the internal split above, tested on the same 84 project images the
    // Python side's Stage 0 uses - a genuine head-to-head between this
    // file's classifiers and task2_combined.py's.
    evaluateWithExternalTraining("TSRD_external/task1_combined_features.csv", csvPath);

    cout << "\nPress any key on an OpenCV window to finish." << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}
