package org.example;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.util.Random;

public class QLearningExample {
    static final int GRID_SIZE = 10;
    static final int NUM_EPISODES = 1000;
    static final double LEARNING_RATE = 0.1;
    static final double DISCOUNT_FACTOR = 0.9;
    static final double EXPLORATION_RATE = 0.1;
    static final int DELAY = 20;  // Delay in milliseconds

    public static void main(String[] args) {
        QLearningAI ai = new QLearningAI(GRID_SIZE * GRID_SIZE, 4, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE);
        double[][] rewards = new double[NUM_EPISODES][2];
        GameStateVisualizer visualizer = new GameStateVisualizer(GRID_SIZE);

        JFrame frame = new JFrame("Q-Learning Performance");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        frame.setSize(600, 800); // Adjusted size

        JPanel chartPanel = createChartPanel(rewards);
        frame.add(chartPanel, BorderLayout.NORTH);
        frame.add(visualizer, BorderLayout.CENTER);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        for (int episode = 0; episode < NUM_EPISODES; episode++) {
            rewards[episode] = runEpisode(ai, visualizer);
            ((ChartPanel) chartPanel).getChart().getXYPlot().datasetChanged(null);
        }
    }

    private static double[] runEpisode(QLearningAI ai, GameStateVisualizer visualizer) {
        GameState state = new GameState(0, 0);
        double totalReward = 0;
        int steps = 0;
        for (int step = 0; step < 100; step++) {
            visualizer.updateState(state);
            try {
                Thread.sleep(DELAY);  // Delay to visualize each step
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            int action = ai.chooseAction(state.toArray());
            GameState nextState = state.move(Action.values()[action]);
            double reward = calculateReward(state, nextState);
            ai.learn(state.toArray(), action, reward, nextState.toArray());
            state = nextState;
            totalReward += reward;
            steps++;
            if (nextState.isGoal()) break;
        }
        return new double[]{steps, totalReward};
    }

    private static double calculateReward(GameState currentState, GameState nextState) {
        int currentDistance = Math.abs(currentState.x - GameState.GOAL_X) + Math.abs(currentState.y - GameState.GOAL_Y);
        int nextDistance = Math.abs(nextState.x - GameState.GOAL_X) + Math.abs(nextState.y - GameState.GOAL_Y);
        if (nextState.isGoal()) return 10;
        return nextDistance < currentDistance ? 0.01 : -0.01;
    }

    private static JPanel createChartPanel(double[][] rewards) {
        XYSeries series = new XYSeries("Reward");
        for (int i = 0; i < rewards.length; i++) {
            series.add(i, rewards[i][1]);
        }
        XYSeriesCollection dataset = new XYSeriesCollection(series);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Q-Learning Performance Over Time",
                "Episode",
                "Total Reward",
                dataset,
                PlotOrientation.VERTICAL,
                false,
                true,
                false
        );

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(500, 300)); // Adjusted size
        return chartPanel;
    }

    static class QLearningAI {
        private double[][] qTable;
        private double learningRate;
        private double discountFactor;
        private double explorationRate;
        private Random random;
        private int numStates;
        private int numActions;

        public QLearningAI(int numStates, int numActions, double learningRate, double discountFactor, double explorationRate) {
            this.numStates = numStates;
            this.numActions = numActions;
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
            this.explorationRate = explorationRate;
            this.qTable = new double[numStates][numActions];
            this.random = new Random();
        }

        public int chooseAction(double[] state) {
            if (random.nextDouble() < explorationRate) {
                return random.nextInt(numActions);
            } else {
                int stateIndex = stateToIndex(state);
                return argMax(qTable[stateIndex]);
            }
        }

        public void learn(double[] state, int action, double reward, double[] nextState) {
            int stateIndex = stateToIndex(state);
            int nextStateIndex = stateToIndex(nextState);
            double qValue = qTable[stateIndex][action];
            double maxNextQValue = max(qTable[nextStateIndex]);
            qTable[stateIndex][action] = qValue + learningRate * (reward + discountFactor * maxNextQValue - qValue);
        }

        private int stateToIndex(double[] state) {
            for (int i = 0; i < state.length; i++) {
                if (state[i] == 1.0) {
                    return i;
                }
            }
            return -1;
        }

        private int argMax(double[] values) {
            int maxIndex = 0;
            for (int i = 1; i < values.length; i++) {
                if (values[i] > values[maxIndex]) {
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        private double max(double[] values) {
            double maxValue = values[0];
            for (double value : values) {
                if (value > maxValue) {
                    maxValue = value;
                }
            }
            return maxValue;
        }
    }

    static class GameState {
        int x, y;
        static final int GOAL_X = 9, GOAL_Y = 9;

        GameState(int x, int y) {
            this.x = x;
            this.y = y;
        }

        boolean isGoal() {
            return x == GOAL_X && y == GOAL_Y;
        }

        GameState move(Action action) {
            int newX = x, newY = y;
            switch (action) {
                case UP:
                    newY = Math.max(y - 1, 0);
                    break;
                case DOWN:
                    newY = Math.min(y + 1, GRID_SIZE - 1);
                    break;
                case LEFT:
                    newX = Math.max(x - 1, 0);
                    break;
                case RIGHT:
                    newX = Math.min(x + 1, GRID_SIZE - 1);
                    break;
            }
            return new GameState(newX, newY);
        }

        double[] toArray() {
            double[] state = new double[GRID_SIZE * GRID_SIZE];
            state[y * GRID_SIZE + x] = 1.0;
            return state;
        }
    }

    enum Action {
        UP, DOWN, LEFT, RIGHT
    }

    static class GameStateVisualizer extends JPanel {
        private static final int CELL_SIZE = 30; // Adjusted cell size
        private GameState state;
        private int gridSize;

        public GameStateVisualizer(int gridSize) {
            this.gridSize = gridSize;
            setPreferredSize(new Dimension(gridSize * CELL_SIZE, gridSize * CELL_SIZE));
        }

        public void updateState(GameState state) {
            this.state = state;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    g.drawRect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }

            if (state != null) {
                g.setColor(Color.RED);
                g.fillRect(state.x * CELL_SIZE, state.y * CELL_SIZE, CELL_SIZE, CELL_SIZE);

                g.setColor(Color.GREEN);
                g.fillRect(GameState.GOAL_X * CELL_SIZE, GameState.GOAL_Y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
    }
}
