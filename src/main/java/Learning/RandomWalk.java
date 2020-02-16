package Learning;

import Model.Simulation;

import javax.swing.*;
import java.io.Serializable;
import java.util.*;

public class RandomWalk extends SubGoalController implements Serializable {

    private double[] costs;

    public RandomWalk(int nrAgents){
        super(nrAgents);
    }

    protected void train(boolean saveMLP, boolean finalIter){
        costs = new double[3];
        for(int test = 0; test < defGenerationSize(); test++){

            resetSimulation();

            model.start();

            int[] modelCosts = fit.totalCosts(model);
            costs[0] += modelCosts[0];
            costs[1] += modelCosts[1];
            costs[2] += modelCosts[2];

        }

        costs[0] /= defGenerationSize();
        costs[1] /= defGenerationSize();
        costs[2] /= defGenerationSize();
    }

    public void test(){
        costs = new double[3];
        for(int i = 0; i < sizeFinalComparison; i++){
            resetSimulation();

            model.start();

            int[] modelCosts = fit.totalCosts(model);
            costs[0] += modelCosts[0];
            costs[1] += modelCosts[1];
            costs[2] += modelCosts[2];

        }

        costs[0] /= sizeFinalComparison;
        costs[1] /= sizeFinalComparison;
        costs[2] /= sizeFinalComparison;
    }

    @Override
    protected void resetSimulation(){
        model = new Simulation(this,use_gui,randSeed, nrAgents);
        resetSubGoals();
    }

    protected void initRL(){
    }


    protected double[] getOutput(double[] input){

        double[] newOutput = {new Random().nextFloat()*9};
        return newOutput;
    }
    protected double[] getCost(){
        return costs;
    }

    protected String defRLMethod() {return "Random_Walk";}

    protected int defGenerationSize() {
        return 50;
    }

}
