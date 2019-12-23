package Learning.CoSyNe;

import Learning.CoSyNe.*;
import Learning.DeepQ.DeepQLearner;
import Learning.Features;
import Learning.Fitness;
import Learning.SubGoalController;
import Model.Agent;
import Model.Simulation;
import Learning.OffsetFeatures;
import Navigation.SubGoal;
import View.MainFrame;
import org.neuroph.util.TransferFunctionType;

import javax.swing.*;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import static java.lang.Double.NaN;

/**
 * SubGoalLearning is the class which learns at what distances the subgoals should be placed.
 * While it is a RLcontroller, we will not be using it as such, since we're not picking actions but defining subgoals
 */
public class SubGoalLearning extends CoSyNe  {
    public SubGoalLearning(){
        super();
        //model = new Simulation(false);  //Not Simulation(this), since we don't pick the individual moves
        //train();
    }
//
//    protected void initSubGoalOrder(){ //TODO: If rand float< exploreRate -> make random dist array, otherwise use order of activations
//        for (Agent a:model.getAgents()){
//            HashMap<String, List<SubGoalController.IndexActLink>> activationMap = new HashMap<>();
//            for (String s:subGoalKeys){
//                mlp.setInput(getInput(s, a));
//                mlp.calculate();
//                double[] outputSet = mlp.getOutput();
//                List<SubGoalController.IndexActLink> outputList = determineOrder(outputSet);
//
//                activationMap.put(s, outputList);
//            }
//            subGoalActivation.put(a, activationMap);
//        }
//        updateDistMap(subGoalActivation);
//    }

    /**
     * The original testMLP assume that this is the RL controller, but that's not the case.
     * We copied that code and changed some things around to fit the task.
     */
    @Override
    protected void testMLP(){

        JFrame frame;
        if(use_gui){
            frame = createMainFrame();
        }

        if ((debugging)&&use_gui){
            //model.applyUpdates();
            sleep(500);
            //SGC.screenshot(run, iter);
        }

        if (debugging){
            printFinalDistMap();
            // printGoalToCoastMap();
        }

        model.start();



        //model.setSubGoals(dist);
        //System.out.println(Arrays.toString( model.getSubGoals()));
        double fitness = getFitness();

        for(int layer = 0; layer < weightBags.size(); layer++){
            for(int neuron = 0; neuron < weightBags.get(layer).size(); neuron++){
                for(int weight = 0; weight < weightBags.get(layer).get(neuron).size(); weight++){
                    WeightBag bag = weightBags.get(layer).get(neuron).get(weight);
                    bag.updateFitness(fitness);
                }
            }
        }
        mean_perfomance += fitness;
        if(best_performance == null || fitness < best_performance){
            best_performance = fitness;
        }
        if(ultimate_performance == null || getFitness() < ultimate_performance){    //take screenshot
            ultimate_performance = fitness;
        }

        if (debugging){
            printFinalDistMap();
        }

        if (use_gui){
            disposeMainFrame(frame);
        }
    }

    protected String defRLMethod() {return "CoSyNe";}

    @Override
    protected void resetSimulation() {
        model = new Simulation(this, use_gui, randSeed);
        resetSubGoals();
    }

    /**
     * we dont use this, since we're not acting as the RLcontroller
     * @param action
     * @param a
     */
    @Override
    protected void performAction(int action, Agent a) {

    }

    @Override
    protected int defN_generations() {
        return 20;
    }

    /**
     * Since inputs are only defined by angle for now, complexity it minimal
     * @return
     */
    @Override
    protected int[] defHiddenLayers() {
        int[] hl = {3};
        return hl;
    }

    /**
     * Only 1 output, from which the value is translated to an offset
     */
    @Override
    protected int defN_outputs() {
        return outputNeurons;
    }

    @Override
    protected int defBagSize() {
        return 10;
    } //TODO: look into effect of bag size

    @Override
    protected int defGenerationSize() {
        return defBagSize()*5;
    }

    @Override
    protected float defAlpha() {
        return 0.05f;
    }

    @Override
    protected int defN_children() {
        return 5;
    }

    @Override
    protected double getFitness() {
        int[] costArr = getCost();

        return costArr[0]+costArr[1]+costArr[2];
    }

    @Override
    protected int defWeightSpread(){
        return 3;
    }

    @Override
    /**
     * Use a sigmoid, since the output from 0-1 is scaled to be center-border.
     */
    protected TransferFunctionType defTransferFunction() {
        return TransferFunctionType.SIGMOID;
    }


    @Override
    protected double defCertainty(){
        return 1;
    }
}
