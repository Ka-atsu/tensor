import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Papa from "papaparse";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import seedrandom from "seedrandom";

// Set a seed for random number generation
seedrandom("my-seed", { global: true });

const PurchaseForecast = () => {
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [jsonData, setJsonData] = useState(null);
  const [productMap, setProductMap] = useState({});
  const [startDate, setStartDate] = useState("");
  const [selectedProduct, setSelectedProduct] = useState(""); // State for selected product

  // Handle CSV file upload
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          const parsedData = results.data.map((row) => ({
            sales_date: row.sales_date,
            product_description: row.product_description,
            quantity_sold: row.quantity_sold,
          }));
          setData(parsedData);
          setJsonData(parsedData);

          // Automatically train the model after parsing the data
          handleTrainModel(parsedData);
        },
        error: (error) => {
          console.error("Error parsing CSV:", error);
        },
      });
    }
  };

  const preprocessData = (data) => {
    const newProductMap = {};
    let productIndex = 0;

    const processedData = data
      .map((entry) => {
        if (!entry.sales_date || isNaN(entry.quantity_sold)) {
          console.warn("Invalid entry:", entry);
          return null;
        }

        const dateParts = entry.sales_date.split(" ")[0].split("/");
        const month = parseInt(dateParts[0], 10);
        const year = parseInt(dateParts[2], 10);

        if (!(entry.product_description in newProductMap)) {
          newProductMap[entry.product_description] = productIndex++;
        }
        const productDescriptionEncoded = newProductMap[entry.product_description];

        return {
          month: year * 12 + month,
          product_description: productDescriptionEncoded,
          quantity_sold: entry.quantity_sold,
        };
      })
      .filter(Boolean);

    const quantities = processedData.map((d) => d.quantity_sold);
    const maxQuantity = Math.max(...quantities);
    const minQuantity = Math.min(...quantities);

    const range = maxQuantity - minQuantity || 1;
    const normalizedData = processedData.map((d) => ({
      ...d,
      quantity_sold: (d.quantity_sold - minQuantity) / range,
    }));

    setProductMap(newProductMap);

    return { normalizedData, maxQuantity, minQuantity };
  };

  const trainModel = async (normalizedData) => {
    const xs = tf.tensor2d(
      normalizedData.map((d) => [
        Math.sin((2 * Math.PI * (d.month % 12)) / 12),
        Math.cos((2 * Math.PI * (d.month % 12)) / 12),
        d.product_description,
      ])
    );
    const ys = tf.tensor1d(normalizedData.map((d) => d.quantity_sold));

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: "relu", inputShape: [3] }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ optimizer: "adam", loss: "meanSquaredError" });

    await model.fit(xs, ys, {
      epochs: 100,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch}: Loss = ${logs.loss}`);
        },
      },
    });

    return model;
  };

  const predictSales = async (model, maxQuantity, minQuantity) => {
    const futurePredictions = [];
    const [year, month] = startDate.split("-").map(Number);

    // Array of month names
    const monthNames = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];

    for (let i = 0; i < 6; i++) {
      // Calculate the next month and year
      const nextMonth = (month + i - 1) % 12 + 1; // This will give you the correct 1-indexed month
      const nextYear = year + Math.floor((month + i - 1) / 12);

      if (selectedProduct) {
        const productIndex = productMap[selectedProduct];
        const sinMonth = Math.sin((2 * Math.PI * (nextMonth - 1)) / 12); // Adjust for 0-indexed month
        const cosMonth = Math.cos((2 * Math.PI * (nextMonth - 1)) / 12); // Adjust for 0-indexed month

        const inputTensor = tf.tensor2d([[sinMonth, cosMonth, productIndex]]);
        const predictionTensor = model.predict(inputTensor);
        const normalizedPrediction = predictionTensor.dataSync()[0];

        const quantitySold = normalizedPrediction * (maxQuantity - minQuantity) + minQuantity;

        // Use monthNames to get the correct month name
        futurePredictions.push({
          sales_date: `${monthNames[nextMonth - 1]} ${nextYear}`, // Format as "Month Year"
          product_description: selectedProduct,
          quantity_sold: quantitySold,
        });
      }
    }

    setPredictions(futurePredictions);
  };

  const handleTrainModel = async (parsedData) => {
    setLoading(true);
    const { normalizedData, maxQuantity, minQuantity } = preprocessData(parsedData);
    const model = await trainModel(normalizedData);
    await predictSales(model, maxQuantity, minQuantity);
    setLoading(false);
  };

  useEffect(() => {
    if (selectedProduct && data.length > 0) {
      // Find the earliest date for the selected product
      const productData = data.filter(
        (item) => item.product_description === selectedProduct
      );
      if (productData.length > 0) {
        const earliestDate = productData.reduce((earliest, current) => {
          const currentDate = new Date(current.sales_date);
          return currentDate < earliest ? currentDate : earliest;
        }, new Date(productData[0].sales_date));

        // Format the date as YYYY-MM
        const formattedDate = `${earliestDate.getFullYear()}-${String(
          earliestDate.getMonth() + 1
        ).padStart(2, "0")}`;
        setStartDate(formattedDate);
      }
    }
  }, [selectedProduct, data]);

  return (
    <div>
      <h1>Purchase Forecast</h1>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <select
        value={selectedProduct}
        onChange={(e) => setSelectedProduct(e.target.value)}
      >
        {Object.keys(productMap).map((product) => (
          <option key={product} value={product}>
            {product}
          </option>
        ))}
      </select>
      <input
        type="text"
        value={startDate}
        onChange={(e) => setStartDate(e.target.value)}
        placeholder="YYYY-MM"
      />
      <button
        onClick={() => handleTrainModel(data)}
        disabled={loading || !startDate || !selectedProduct}
      >
        {loading ? "Training..." : "Train Model"}
      </button>
      {predictions.length > 0 && (
        <LineChart width={800} height={400} data={predictions}>
          <XAxis
            dataKey="sales_date"
            textAnchor="end"
            tick={{ fontSize: 12 }}
            interval={0}
          />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <CartesianGrid strokeDasharray="3 3" />
          <Line type="monotone" dataKey="quantity_sold" stroke="#8884d8" />
        </LineChart>
      )}
    </div>
  );
};

export default PurchaseForecast;