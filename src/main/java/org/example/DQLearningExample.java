package org.example;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;

public class DQLearningExample {
    static final int GRID_SIZE = 10;
    static final int NUM_EPISODES = 1000;
    static final double LEARNING_RATE = 0.001;
    static final double DISCOUNT_FACTOR = 0.9;
    static final double EXPLORATION_RATE = 0.1;
    static final int DELAY = 100;  // Delay in milliseconds

    public static void main(String[] args) {
        DQLearningAI ai = new DQLearningAI(GRID_SIZE * GRID_SIZE, 4, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE);
        double[][] rewards = new double[NUM_EPISODES][2];
        GameStateVisualizer visualizer = new GameStateVisualizer(GRID_SIZE);

        JFrame frame = new JFrame("DQN Performance");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());

        JPanel chartPanel = createChartPanel(rewards);
        frame.add(chartPanel, BorderLayout.NORTH);
        frame.add(visualizer, BorderLayout.CENTER);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        for (int episode = 0; episode < NUM_EPISODES; episode++) {
            rewards[episode] = runEpisode(ai, visualizer);
            ((ChartPanel) chartPanel).getChart().getXYPlot().datasetChanged(null);
        }
    }

    private static double[] runEpisode(DQLearningAI ai, GameStateVisualizer visualizer) {
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
            double reward = nextState.isGoal() ? 10 : -1;  // Smaller reward for reaching the goal
            ai.learn(state.toArray(), action, reward, nextState.toArray());
            state = nextState;
            totalReward += reward;
            steps++;
            if (nextState.isGoal()) break;
        }
        return new double[] {steps, totalReward};
    }

    private static JPanel createChartPanel(double[][] rewards) {
        XYSeries series = new XYSeries("Reward");
        for (int i = 0; i < rewards.length; i++) {
            series.add(rewards[i][0], rewards[i][1]);
        }
        XYSeriesCollection dataset = new XYSeriesCollection(series);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "DQN Performance Over Time",
                "Episode",
                "Total Reward",
                dataset,
                PlotOrientation.VERTICAL,
                false,
                true,
                false
        );

        return new ChartPanel(chart);
    }
}

class GameState {
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
        int new
