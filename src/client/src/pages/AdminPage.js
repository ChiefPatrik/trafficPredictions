import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto'; // This is required for Chart.js 3.x

const citiesCoordinates = {
  "Ljubljana": { "latitude": 46.07061503201507, "longitude": 14.577867970254866 },
  "Maribor": { "latitude": 46.68255971126553, "longitude": 15.65138919777721 },
  "Sl_Konjice": { "latitude": 46.25413437290015, "longitude": 15.302557315050453 },
  "Postojna": { "latitude": 45.93134443045748, "longitude": 14.270708378492925 },
  "Vransko": { "latitude": 46.174640576447764, "longitude": 14.804130481638964 },
  "Pomurska": { "latitude": 46.52351975291412, "longitude": 16.44175950632071 },
  "Kozina": { "latitude": 45.60742223894982, "longitude": 13.927767896289717 }
};

const filteredParams = [
  'epochs',
  'opt_name',
  'opt_learning_rate',
  'batch_size',
  'validation_split',
  'shuffle'
];

function AdminPage() {
  const [selectedCity, setSelectedCity] = useState('');
  const [carsParams, setCarsParams] = useState(null);
  const [carsMetrics, setCarsMetrics] = useState(null);
  const [carsEvalMetrics, setCarsEvalMetrics] = useState(null);
  const [speedParams, setSpeedParams] = useState(null);
  const [speedMetrics, setSpeedMetrics] = useState(null);
  const [speedEvalMetrics, setSpeedEvalMetrics] = useState(null);

  const handleDropdownChange = async (e) => {
    const city = e.target.value;
    setSelectedCity(city);

    if (city !== '') {
      try {
        const response = await fetch('http://localhost:3001/traffic/evaluation/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            data: [
              { region: city }
            ]
          })
        });

        if (response.ok) {
          const data = await response.json();
          const cars_params = data["evaluation"]["cars_evaluation"]["params"]
          const cars_metrics = data["evaluation"]["cars_evaluation"]["metrics"]
          const cars_eval_metrics = data["evaluation"]["cars_evaluation"]["eval_metrics"]
          const speed_params = data["evaluation"]["speed_evaluation"]["params"]
          const speed_metrics = data["evaluation"]["speed_evaluation"]["metrics"]
          const speed_eval_metrics = data["evaluation"]["speed_evaluation"]["eval_metrics"]
          setCarsParams(cars_params);
          setCarsMetrics(cars_metrics);
          setCarsEvalMetrics(cars_eval_metrics);
          setSpeedParams(speed_params);
          setSpeedMetrics(speed_metrics);
          setSpeedEvalMetrics(speed_eval_metrics);

        } else {
          console.error('Failed to fetch evaluation data');
        }
      } catch (error) {
        console.error('Error:', error);
      }
    }
  };

  const lineChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'MSE vs Runs',
      },
    },
  };

  const carsLineChartData = {
    labels: carsMetrics ? carsMetrics.map((_, index) => `Run ${index + 1}`) : [],
    datasets: [
      {
        label: 'MSE',
        data: carsMetrics ? carsMetrics.map(metric => metric.mse) : [],
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
      },
    ],
  };

  const speedLineChartData = {
    labels: speedMetrics ? speedMetrics.map((_, index) => `Run ${index + 1}`) : [],
    datasets: [
      {
        label: 'MSE',
        data: speedMetrics ? speedMetrics.map(metric => metric.mse) : [],
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
      },
    ],
  };



  return (
    <div className="flex flex-col items-center h-screen w-1/2 mx-auto flex pt-24">
      <h1 className="text-3xl font-bold mb-8">Evaluation Dashboard</h1>
      <div>
        <div>
          <label htmlFor="city" className="block flex justify-center text-gray-700">Select Region</label>
          <select
            id="city"
            value={selectedCity}
            onChange={handleDropdownChange}
            className="mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          >
            <option value="">-- Select a region --</option>
            {Object.keys(citiesCoordinates).map((city) => (
              <option key={city} value={city}>
                {city}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* <div className="flex flex-row justify-center w-full">
        <div className="flex flex-col items-center w-1/2 px-4"> */}
          {carsParams && (       
            <div className="mt-8 w-full max-w-lg">
              <h1 className="text-3xl flex justify-center font-bold ">"Num_of_cars" model</h1>
              <h2 className="text-xl flex justify-center font-bold mt-6 mb-4">Parameters</h2>
              <pre className="bg-gray-100 p-4 rounded-md">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead>
                    <tr>
                      <th className="px-4 py-2 border">Parameter</th>
                      <th className="px-4 py-2 border">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredParams.map((key) => (
                      <tr key={key}>
                        <td className="border px-4 py-2">{key}</td>
                        <td className="border px-4 py-2">{carsParams[key]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </pre>
            </div>
          )}
          {carsEvalMetrics && (
            <div className="mt-8 w-full max-w-lg">
              <h2 className="text-xl flex justify-center font-bold mb-4">Evaluation Metrics</h2>
              <pre className="bg-gray-100 p-4 rounded-md">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead>
                    <tr>
                      <th className="px-4 py-2 border">MAE</th>
                      <th className="px-4 py-2 border">MSE</th>
                      <th className="px-4 py-2 border">EVS</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border px-4 py-2">{carsEvalMetrics["mae"]}</td>
                      <td className="border px-4 py-2">{carsEvalMetrics["mse"]}</td>
                      <td className="border px-4 py-2">{carsEvalMetrics["evs"]}</td>
                    </tr>
                  </tbody>
                </table>
              </pre>
            </div>
          )}
          {carsMetrics && (
            <div className="mt-8 w-full max-w-lg">
              <h2 className="text-xl flex justify-center font-bold mb-2">MSE Latest 25 Train Runs</h2>
              <Line data={carsLineChartData} options={lineChartOptions} />
            </div>
          )}
        {/* </div>
        <div className="flex flex-col items-center w-1/2 px-4"> */}
          {speedParams && (
            <div className="mt-8 w-full max-w-lg">
              <h1 className="text-3xl flex justify-center font-bold">"Avg_speed" model</h1>
              <h2 className="text-xl flex justify-center font-bold mt-6 mb-4">Parameters</h2>
              <pre className="bg-gray-100 p-4 rounded-md">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead>
                    <tr>
                      <th className="px-4 py-2 border">Parameter</th>
                      <th className="px-4 py-2 border">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredParams.map((key) => (
                      <tr key={key}>
                        <td className="border px-4 py-2">{key}</td>
                        <td className="border px-4 py-2">{speedParams[key]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </pre>
            </div>
          )}
          {speedEvalMetrics && (
            <div className="mt-8 w-full max-w-lg">
              <h2 className="text-xl flex justify-center font-bold mb-4">Evaluation Metrics</h2>
              <pre className="bg-gray-100 p-4 rounded-md">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead>
                    <tr>
                      <th className="px-4 py-2 border">MAE</th>
                      <th className="px-4 py-2 border">MSE</th>
                      <th className="px-4 py-2 border">EVS</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border px-4 py-2">{speedEvalMetrics["mae"]}</td>
                      <td className="border px-4 py-2">{speedEvalMetrics["mse"]}</td>
                      <td className="border px-4 py-2">{speedEvalMetrics["evs"]}</td>
                    </tr>
                  </tbody>
                </table>
              </pre>
            </div>
          )}
          {speedMetrics && (
            <div className="mt-8 w-full max-w-lg">
              <h2 className="text-xl flex justify-center font-bold mb-2">MSE Latest 25 Train Runs</h2>
              <Line data={speedLineChartData} options={lineChartOptions} />
            </div>
          )}
        {/* </div>
      </div> */}
    </div>
  );
}

export default AdminPage;
