USE diabetes_db;

SET hive.execution.engine=mr;

-- 1. Total valid records
SELECT COUNT(*)
FROM diabetes_raw
WHERE ID IS NOT NULL;

-- 2. Diabetes class distribution
SELECT Diabetes_binary, COUNT(*)
FROM diabetes_raw
WHERE ID IS NOT NULL
GROUP BY Diabetes_binary;

-- 3. BMI and general health analysis
SELECT
    Diabetes_binary,
    ROUND(AVG(BMI), 2) AS average_bmi,
    ROUND(AVG(GenHlth), 2) AS average_general_health
FROM diabetes_raw
WHERE ID IS NOT NULL
GROUP BY Diabetes_binary;

-- 4. Blood pressure and cholesterol analysis
SELECT
    Diabetes_binary,
    ROUND(AVG(HighBP) * 100, 2) AS high_bp_percentage,
    ROUND(AVG(HighChol) * 100, 2) AS high_chol_percentage
FROM diabetes_raw
WHERE ID IS NOT NULL
GROUP BY Diabetes_binary;

-- 5. Diabetes percentage by age category
SELECT
    Age,
    COUNT(*) AS total_people,
    ROUND(AVG(Diabetes_binary) * 100, 2) AS diabetes_percentage
FROM diabetes_raw
WHERE ID IS NOT NULL
GROUP BY Age
ORDER BY Age;
