package Learning.CoSyNe;

import Learning.SubGoalController;
import Model.Agent;
import Model.Simulation;
import View.MainFrame;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;

public abstract class CoSyNe extends SubGoalController {
    private List<Integer> MLP_shape;
    //layer, neuron, weight
    protected List<List<List<WeightBag>>> weightBags;
    protected List<FitnessToMLP> savedMLPList;
    protected MultiLayerPerceptron mlp;
    protected int outputNeurons = 1;
    protected Double best_performance = null;
    protected Double ultimate_performance = null;
    protected double mean_perfomance;
    protected Double mean_confidence = null;
    protected Integer conf_counter = null;
    protected int generation;
    protected double[] costs;

    public CoSyNe(int nrAgents){
        super(nrAgents);
    }

    public CoSyNe(){
        super(1);
    }

    protected void initRL(){
        MLP_shape = new ArrayList<>();
        savedMLPList = new ArrayList<>();
        costs = new double[3];

        outputNeurons = 1;

        MLP_shape.add(getInputSet("WW", model.getAgents().get(0)).length);
        for(int i = 0; i < defHiddenLayers().length; i++){
            MLP_shape.add(defHiddenLayers()[i]);
        }
        MLP_shape.add(defN_outputs());

        initializeBags();
    }

    /**
     * The overall generation loop including creating MLPs, testing them, and breeding them
     */
    protected void train(boolean saveMLP, boolean finalIter){ //Implement generations
        mean_perfomance = 0;
        costs = new double[3];
        for(int test = 0; test < defGenerationSize(); test++){

            resetSimulation();

            createMLP();

            testMLP();

            int[] modelCosts = fit.totalCosts(model);
            costs[0] += modelCosts[0];
            costs[1] += modelCosts[1];
            costs[2] += modelCosts[2];

            if (finalIter){
                savedMLPList.add(new FitnessToMLP(getFitness(), mlp));
            }
        }
        if (finalIter) {
            Collections.sort(savedMLPList);
        }
        costs[0] /= defGenerationSize();
        costs[1] /= defGenerationSize();
        costs[2] /= defGenerationSize();

        mean_perfomance /= defGenerationSize();
        printPerformance();
        best_performance = null;
        breed();
    }

    protected void test(){ //Implement generations
        mean_perfomance = 0;
        costs = new double[3];
        for(int i = 0; i < sizeFinalComparison; i++){
            resetSimulation();
            mlp = savedMLPList.get(i).mlp;
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


    /**
     * Print performance. Can be nice to override if you want extra information
     */
    protected void printPerformance(){
        System.out.println("Best performance: " + best_performance);
        System.out.println("Mean performance: " + mean_perfomance);
    }

    protected double[] getCost(){
        return costs;
    }

    /**
     * Form an MLP by pulling random weights out of the weightbags.
     */
    protected void createMLP(){
        mlp = new MultiLayerPerceptron(MLP_shape, defTransferFunction());
        for (int layer = 0; layer < mlp.getLayersCount(); layer ++) {
            for (int neuron = 0; neuron < mlp.getLayerAt(layer).getNeuronsCount(); neuron++) {
                for (int weight = 0; weight < mlp.getLayerAt(layer).getNeuronAt(neuron).getWeights().length; weight++) {
                    mlp.getLayerAt(layer).getNeuronAt(neuron).getInputConnections().get(weight).setWeight(weightBags.get(layer).get(neuron).get(weight).randomWeight());
                }
            }
        }
    }

    /**
     * Subject the MLP to the simulation so we can establish its fitness.
     */
    protected void testMLP(){


        for(int layer = 0; layer < weightBags.size(); layer++){
            for(int neuron = 0; neuron < weightBags.get(layer).size(); neuron++){
                for(int weight = 0; weight < weightBags.get(layer).get(neuron).size(); weight++){
                    WeightBag bag = weightBags.get(layer).get(neuron).get(weight);
                    bag.updateFitness(getFitness());
                }
            }
        }
        mean_perfomance += getFitness();
        if(best_performance == null || getFitness() < best_performance){
            best_performance = getFitness();
        }
        if(ultimate_performance == null || getFitness() < ultimate_performance){    //take screenshot
            model = new Simulation(this, nrAgents);

            JFrame f = new MainFrame(model);
            model.start();
            sleep(1000);
            screenshot(0, (int) getFitness());
            ultimate_performance = getFitness();
            f.dispose();
        }
    }

    /**
     * Takes a screenshot. Generation and i are only used to define the name of the file.
     * @param generation
     * @param i
     */
    protected void screenshot(int generation, int i){
        Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
        try {
            BufferedImage capture = new Robot().createScreenCapture(screenRect);
            ImageIO.write(capture, "bmp", new File("./screenshot_g"+ generation+"_i_"+i+".bmp"));

        }catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

    private class FitnessToMLP implements Comparable<FitnessToMLP> {
        private final Double fitness;
        private final MultiLayerPerceptron mlp;

        public FitnessToMLP(double fitness, MultiLayerPerceptron mlp) {
            this.fitness = fitness;
            this.mlp = mlp;
        }

        public double fitness() { return fitness; }
        public MultiLayerPerceptron mlp() {return mlp; }

        public int compareTo(FitnessToMLP o) {
            return this.fitness.compareTo(o.fitness);
        }
    }


    /**
     * Tell all bags to breed
     */
    protected void breed(){
        for(int layer = 0; layer < weightBags.size(); layer++){
            for(int neuron = 0; neuron < weightBags.get(layer).size(); neuron++) {
                for (int weight = 0; weight < weightBags.get(layer).get(neuron).size(); weight++) {
                    WeightBag bag = weightBags.get(layer).get(neuron).get(weight);
                    bag.breed(defN_children());
                }
            }
        }
    }

    /**
     * Creates all the weight bags.
     */
    protected void initializeBags(){
        weightBags = new ArrayList<>();
        mlp = new MultiLayerPerceptron(MLP_shape);
        int bagSize = defBagSize();
        for (int layer = 0; layer < mlp.getLayersCount(); layer ++) {
            weightBags.add(new ArrayList<>());
            for(int neuron = 0; neuron < mlp.getLayerAt(layer).getNeuronsCount(); neuron++){
                weightBags.get(layer).add(new ArrayList<>());
                for(int weight = 0; weight < mlp.getLayerAt(layer).getNeuronAt(neuron).getWeights().length; weight++){
                    weightBags.get(layer).get(neuron).add(new WeightBag(bagSize, defAlpha(), defWeightSpread()));
                }
            }
        }
    }

//    /**
//     * Extracts an action integer from the MLPs output. Using SoftMax
//     * @param a
//     */
//    @Override
//    public void pickAction(Agent a) {
//        super.pickAction(a);
//        double[] outputs = getOutput(getInput());
//
//
//        //We apply softMax
//        double sum = 0;
//
//        for(int i = 0; i< outputs.length; i++){
//            sum = sum + Math.exp(outputs[i]/ defCertainty());
//        }
//
//        double rand = new Random().nextDouble();
//
//        double step = 0;
//        int chosen_action = -1;
//        while(chosen_action < outputs.length && step < rand){
//            chosen_action++;
//            step += Math.exp(outputs[chosen_action]/defCertainty())/sum;
//        }
//
//        if(mean_confidence == null){
//            mean_confidence = new Double(0);
//        }
//        if(conf_counter == null){
//            conf_counter = new Integer(0);
//        }
//        //Log the confidence so that they become printable
//        mean_confidence = mean_confidence + Math.exp(outputs[chosen_action]/defCertainty())/sum;
//        conf_counter++;
//    }


    protected double[] getOutput(double[] input){

        mlp.setInput(input);
        mlp.calculate();
        double[] output = mlp.getOutput();
        double[] newOutput = {output[0]*10};
        return newOutput;
    }


    /**
     * Define how an action i should be performed
     * @param action
     */
    protected abstract void performAction(int action, Agent a);

    /**
     * Define the number of generations the CoSyNe needs to learn for
     * @return the number of generations wanted
     */
    protected abstract int defN_generations();

    /**
     * Define the shape of hidden layer(s).
     * @return An array of ints representing the number of neurons in each hidden layer
     */
    protected abstract int[] defHiddenLayers();

    /**
     * Define the number of outputs the MLP should be able to produce. Make sure this corresponds with pickAction
     * @return The number of possible actions the MLP should be able to take
     */
    protected abstract int defN_outputs();

    /**
     * Specify the number of weights in each bag
     * @return Number of weights in each bag
     */
    protected abstract int defBagSize();

    /**
     * Specify the number of MLPs which should be created&tested in each generation.
     * A larger value will more accuractely assess the performance of each weight in a generation, but take more time
     * A value of 1-5x bagsize seems reasonable
     * @return Number of MLPs created in each generation
     */
    protected abstract int defGenerationSize();

    /**
     * Specify the learning rate at which the fitness of a weight is evaluated. This mimics the discount rate discussed in the Reinforcement Learning book.
     * Alpha at 1 will make the performance of a weight based only on the most recent trial.
     * Alpha at 0 will make the performance of a weight based only on the first trial.
     * Alpha at 0.05 is common, where the last 20 trials mostly determine the performance.
     * @return
     */
    protected abstract float defAlpha();

    /**
     * Specify the number of children which should be generated on each generation
     * @return
     */
    protected abstract int defN_children();
//
//    /**
//     * Get the inputs to the MLP from a model
//     * @return A list of doubles extracted as meaningful features from the model
//     */
//    protected abstract double[] getInput();

    /**
     * Get the performance the MLP delivered from the model
     * @return A double representing the model's performance
     */
    protected abstract double getFitness();

    /**
     * DefWeightSpread determines the range the weigths are spawned in [-x, x].
     * Due to the SoftMax output, a larger range makes the SoftMax less stochastic, while a lower range makes it more.
     * @return
     */
    protected abstract int defWeightSpread();

    /**
     * Pick a transferFunction for the MLP. Reasonable options are:
     * TransferFunctionType.RECTIFIED (ReLU)
     * TransferFunctionType.SIGMOID
     * @return
     */
    protected abstract TransferFunctionType defTransferFunction();

    /**
     * Certainty determines how stochastic the actions from the MLPs will be.
     * A small value grants low stochasticity (i.e. more certain). A value too small (0.01) results in NaNs.
     * A larger value (1-5) grants high stochasticity.
     *
     * @return
     */
    protected abstract double defCertainty();

}
