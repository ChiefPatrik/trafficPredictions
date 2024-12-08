import { render, screen, fireEvent } from '@testing-library/react';
import PredictionsPage from '../pages/PredictionsPage';

global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ predictions: { num_of_cars: 150, avg_speed: 60 } }),
  })
);

test('predict traffic button calls API', async () => {
  render(<PredictionsPage />);
  const predictButton = screen.getByTestId("predictBtn");

  fireEvent.click(predictButton);

  expect(global.fetch).toHaveBeenCalledTimes(1);
  expect(global.fetch).toHaveBeenCalledWith(
    `${process.env.REACT_APP_PREDICTION_API_URL}/traffic/predict`,
    expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({
        'Content-Type': 'application/json',
      }),
    })
  );

  // Assert labels are updated with the API response
  const carsPrediction = await screen.findByTestId('cars_prediction');
  const speedPrediction = await screen.findByTestId('speed_prediction');

  expect(Number(carsPrediction.textContent)).toBeGreaterThan(0);
  expect(Number(speedPrediction.textContent)).toBeGreaterThan(0);
});
