import React, { useState, useRef } from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import TimePicker from 'react-time-picker';
import 'react-time-picker/dist/TimePicker.css';
import 'react-clock/dist/Clock.css';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-defaulticon-compatibility';
import 'leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css';

const citiesCoordinates = {
  "Ljubljana": { "latitude": 46.07061503201507, "longitude": 14.577867970254866 },
  "Maribor": { "latitude": 46.68255971126553, "longitude": 15.65138919777721 },
  "Sl_Konjice": { "latitude": 46.25413437290015, "longitude": 15.302557315050453 },
  "Postojna": { "latitude": 45.93134443045748, "longitude": 14.270708378492925 },
  "Vransko": { "latitude": 46.174640576447764, "longitude": 14.804130481638964 },
  "Pomurska": { "latitude": 46.52351975291412, "longitude": 16.44175950632071 },
  "Kozina": { "latitude": 45.60742223894982, "longitude": 13.927767896289717 }
};

const PredictionsPage = () => {
  const [selectedCity, setSelectedCity] = useState('');
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);
  const [markerPopupOpen, setMarkerPopupOpen] = useState(false);
  const mapRef = useRef(null);

  const today = new Date();
  const maxDate = new Date();
  maxDate.setDate(today.getDate() + 14);

  // Function to handle marker click
  const handleMarkerClick = (city) => {
    const marker = citiesCoordinates[city];
    if (marker) {
      mapRef.current.setView([marker.latitude, marker.longitude], 12);
      setSelectedCity(city);
      setMarkerPopupOpen(true);
    }
  };

  // Function to handle dropdown menu selection change
  const handleDropdownChange = (e) => {
    const city = e.target.value;
    if (city === '') {
      // Reset map view to its original zoom
      mapRef.current.setView([46.1512, 14.9955], 8);
      setSelectedCity(city);
      setMarkerPopupOpen(false); // Close marker popup
    } else {
      const marker = citiesCoordinates[city];
      if (marker) {
        mapRef.current.setView([marker.latitude, marker.longitude], 12);
        setSelectedCity(city);
        setMarkerPopupOpen(true); // Open marker popup
      }
    }
  };

  
  // Function to format date to match "%Y-%m-%d"
  const formatDate = (date) => {
    const year = date.getFullYear();
    const month = ('0' + (date.getMonth() + 1)).slice(-2);
    const day = ('0' + date.getDate()).slice(-2);
    return `${year}-${month}-${day}`;
  };

  // Function to round time to the nearest hour and add ":00" as minutes
  const formatHour = (time) => {
    const [hours, minutes] = time.split(':');
    let roundedHour = parseInt(hours);
    if (parseInt(minutes) >= 30) {
      roundedHour++;
    }
    const formattedHour = roundedHour < 10 ? `0${roundedHour}` : `${roundedHour}`;
    return `${formattedHour}:00`;
  };  

  // Function to handle "Predict traffic" button click
  const handlePredictTraffic = () => {
    // Format date to match "%Y-%m-%d"
    const formattedDate = selectedDate ? formatDate(selectedDate) : '';

    // Round the time to the nearest hour and add ":00" as minutes
    const formattedHour = selectedTime ? formatHour(selectedTime) : '';

    console.log('Selected city:', selectedCity);
    console.log('Selected date:', formattedDate);
    console.log('Selected time:', formattedHour);

    const payload = {
      data: [
        {
          region: selectedCity,
          date: formattedDate,
          hour: formattedHour
        }
      ]
    };

    // Send POST request
    fetch('http://localhost:3001/traffic/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      console.log('Prediction data:', data);
      document.getElementById('cars_prediction').innerText = data.predictions.num_of_cars + "/h";
      document.getElementById('speed_prediction').innerText = data.predictions.avg_speed + " km/h";

      document.getElementById('cars_prediction_title').classList.remove('invisible');
      document.getElementById('speed_prediction_title').classList.remove('invisible');
      document.getElementById('cars_prediction').classList.remove('invisible');
      document.getElementById('speed_prediction').classList.remove('invisible');
    })
    .catch(error => {
      console.error('Error:', error);
    });
  };


  return (
    <div className="flex flex-col items-center h-screen w-1/2 mx-auto flex pt-24">
      <h1 className="text-3xl font-bold mb-8">Traffic Predictions</h1>
      <div className="z-10 grid grid-cols-3 gap-4 mb-8">
        <div>
          <label htmlFor="city" className="block text-gray-700">Select Region</label>
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
        <div>
          <label htmlFor="date" className="block text-gray-700">Select Date</label>
          <DatePicker
            selected={selectedDate}
            onChange={(date) => setSelectedDate(date)}
            className="mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            placeholderText="Select a date"
            minDate={today}
            maxDate={maxDate}
          />
        </div>
        <div>
          <label htmlFor="time" className="block text-gray-700">Select Time</label>
          <TimePicker
            onChange={setSelectedTime}
            value={selectedTime}
            className="mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            disableClock={false}  // Enable the clock
          />
        </div>
      </div>
      <MapContainer ref={mapRef} center={[46.1512, 14.9955]} zoom={8} style={{ height: '400px', width: '80%', zIndex: 0 }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {Object.entries(citiesCoordinates).map(([city, coords]) => (
          <Marker key={city} position={[coords.latitude, coords.longitude]} eventHandlers={{ click: () => handleMarkerClick(city) }}>
            {markerPopupOpen && <Popup>{city}</Popup>}
          </Marker>
        ))}
      </MapContainer>
      
      <button onClick={handlePredictTraffic} className="mt-8 py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
        Predict traffic
      </button>
      <div className="mt-4"></div>
      <div className="flex justify-center">
        <label id="cars_prediction_title" className="invisible w-32 px-2 py-1 mx-2 ">Number of cars</label>
        <label id="speed_prediction_title" className="invisible w-32 px-2 py-1 mx-2 ">Average speed</label>
      </div>
      <div className="flex justify-center">
        <label id="cars_prediction" className="invisible w-32 px-2 py-1 mx-2 border rounded flex justify-center items-center"></label>
        <label id="speed_prediction" className="invisible w-32 px-2 py-1 mx-2 border rounded flex justify-center items-center"></label>
      </div>
    </div>
  );
};

export default PredictionsPage;
