import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import PredictionsPage from './pages/PredictionsPage';
import AdminPage from './pages/AdminPage';
import Header from './components/Header';

function AppRoutes() {

  return (
    <Routes>
      <Route path="/" element={<PredictionsPage />} />
      <Route path="/admin" element={<AdminPage />} />
    </Routes>
  );
}

function App() {
  return (
    <div className="min-h-screen">
      <Router>
        <Header />
        <div   >  
          {/* className="pt-16" */}
          <AppRoutes />
        </div>
      </Router>
    </div>
  );
}

export default App;
