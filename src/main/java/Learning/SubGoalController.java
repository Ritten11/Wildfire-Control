package Learning;

import Model.Agent;
import Model.Simulation;
import Navigation.OrthogonalSubGoals;
import View.MainFrame;
import org.apache.commons.lang3.time.StopWatch;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class SubGoalController implements Serializable,RLController {

    protected static String RLMethod;
    protected static int nrAgents;

    private final int max_runs = 20;
    protected int sizeFinalComparison = 10;
    protected int run=0;
    protected static int iter=0;
    private static int nrErrors = 0;
    protected int trainingIterations = 500;
    protected int testingIterations = 50;
    protected float explorationRate;
    protected float exploreDiscount = explorationRate/ trainingIterations;

    protected OrthogonalSubGoals subGoals;
    protected Simulation model;

    //Variables needed for debugging:
    protected final static boolean use_gui = false;
    protected final static boolean debugging = false;
    protected final static int timeActionShown = 100;
    protected int showActionFor;
    protected long randSeed = 0;

    protected Agent backup; //If the final agent has died, the MLP still needs an agent to determine the inputVector of the MLP

    //Fields for functionality of navigation and fitness
    protected String algorithm = "Bresenham";
    protected Fitness fit;
    protected Features f;
    private int lowestCost;
    private double[][] costArrTraining;
    private double[][] costArrTesting;
    private StopWatch watch;

    protected HashSet<String> assignedGoals;
    protected HashMap<Agent, HashMap<String, List<IndexActLink>>> subGoalActivation;
    protected final List<String> subGoalKeys = Arrays.asList(new String[]{"WW", "SW", "SS", "SE", "EE", "NE", "NN", "NW"});

    private Map<String,Double> distMap = Stream.of(
            new AbstractMap.SimpleEntry<>("WW", 0.0),
            new AbstractMap.SimpleEntry<>("SW", 0.0),
            new AbstractMap.SimpleEntry<>("SS", 0.0),
            new AbstractMap.SimpleEntry<>("SE", 0.0),
            new AbstractMap.SimpleEntry<>("EE", 0.0),
            new AbstractMap.SimpleEntry<>("NE", 0.0),
            new AbstractMap.SimpleEntry<>("NN", 0.0),
            new AbstractMap.SimpleEntry<>("NW", 0.0))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
//
    private Map<String, InputCost> goalToCostMap= Stream.of(
            new AbstractMap.SimpleEntry<>("WW", new InputCost()),
            new AbstractMap.SimpleEntry<>("SW", new InputCost()),
            new AbstractMap.SimpleEntry<>("SS", new InputCost()),
            new AbstractMap.SimpleEntry<>("SE", new InputCost()),
            new AbstractMap.SimpleEntry<>("EE", new InputCost()),
            new AbstractMap.SimpleEntry<>("NE", new InputCost()),
            new AbstractMap.SimpleEntry<>("NN", new InputCost()),
            new AbstractMap.SimpleEntry<>("NW", new InputCost()))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

    public SubGoalController(int nrAgents){

        RLMethod = defRLMethod();
        this.nrAgents = nrAgents;

        f = new Features();
        fit = new Fitness();
        watch = new StopWatch();


        while (run<max_runs) {
            watch.start();
            run++;
            System.out.println("=====================STARTING RUN #" + run + "==================");
            lowestCost = Integer.MAX_VALUE;
            costArrTraining = new double[trainingIterations][3];
            costArrTesting = new double[testingIterations][3];
            randSeed = 0;
            explorationRate = 0.3f;


            model = new Simulation(false, nrAgents);
            // init subGoals

            initRL();

            for (iter = 0; iter < trainingIterations-sizeFinalComparison; iter++) {
                runIteration(false, false);
            }
            for ( ; iter < trainingIterations-1; iter++) {
                runIteration(true, false);
            }
            runIteration(true, true);
            for (int testIter = 0; testIter<testingIterations; testIter++){
                randSeed++;
                test();
                costArrTesting[testIter]=getCost();
            }
            watch.stop();
            writePerformanceFile();

            if (nrErrors != 0) {
                System.out.println("Total # of errors occurred: " + nrErrors);
            }
            watch.reset();
        }
    }

    private void runIteration(boolean saveMLP, boolean finalIter){
        if (debugging) {
            System.out.println("Current iteration:" + iter + " <-------------------------------");
        }
        randSeed++;
        showActionFor = timeActionShown;

        resetSimulation();

        train(saveMLP, finalIter);
        costArrTraining[iter] = getCost(); //TODO: Fix for CoSyNE approach
        if (explorationRate > 0) {
            explorationRate -= exploreDiscount;
        }
    }



    public void updateDistMap(String key, Agent agent, List<IndexActLink> list){

        if (!(subGoals.isGoalOfAgent(key)||assignedGoals.contains(key))) {
//            if (debugging) {
//                System.out.println("updating goal " + key + " for agent #" + agent.getId());
//            }
            double[] input = getInputSet(key, agent);

            setDistance(list, key);
            double[] in = goalToCostMap.get(key).stateX;
            goalToCostMap.get(key).setStateX(input);
            if (debugging) {
                System.out.println("Input changed to " + Arrays.toString(goalToCostMap.get(key).stateX) + " from " + Arrays.toString(in));
            }

        } else {
            if (debugging) {
                System.out.println("Not updating goal already assigned to/reached by other agent -> ");
            }
        }
    }

    public void updateDistMap(HashMap<Agent, HashMap<String, List<IndexActLink>>> subGoalOrder){
        for (Agent a:model.getAgents()){
            for (String goal : subGoalKeys){
                updateDistMap(goal, a, subGoalOrder.get(a).get(goal));
            }
            subGoals.selectClosestSubGoal(a);
            goalToCostMap.get(subGoals.getAgentGoals().get(a)).setStateX(getInputSet(subGoals.getAgentGoals().get(a),a));
        }
    }

//    public void takeNextAction(Agent a) {
//        if (model.goalsHit<distMap.keySet().size()) {
//            if (a.onGoal()) {
//                assignedGoals.add(subGoals.getAgentGoals().get(a));
//                updateGoalsHit(a);
//                updateDistMap(subGoals.getNextGoal(a), a);
//                subGoals.setNextGoal(a);
//            }
//            String nextAction = a.subGoal.getNextAction();
//            if (nextAction.equals("PathFailed")){
//                subGoals.resetGoal(a);
//                takeNextAction(a);
//                return;
//            } else {
//                a.takeAction(nextAction);
//            }
//            // TODO: This piece of code is ugly as hell, come up with better solution
//            if (model.getAllCells().get(a.getX()).get(a.getY()).isBurning()) {
//                subGoals.removeGoalReached(a);
//                backup = a;
//                if (debugging) {
//                    System.out.println("Nr of Agents: " + model.getAgents().size());
//                }
//            }
//        } else { //Once all goals have been reached, the agent should stop moving as there is no use for it anymore.
//            a.takeAction("Do Nothing");
//        }
//    }

    public String getNextAction(Agent a){
        if (model.goalsHit<distMap.keySet().size()) {
            if (a.onGoal()) {
                assignedGoals.add(subGoals.getAgentGoals().get(a));
                updateGoalsHit(a);

                return "Update sub-goal";
//                updateDistMap(subGoals.getNextGoal(a), a);
//                subGoals.setNextGoal(a);
            }
            String nextAction = a.subGoal.getNextAction();
            if (nextAction.equals("PathFailed")){
                subGoals.resetGoal(a);
                return getNextAction(a);
            } else {
                return  nextAction;
            }
        } else { //Once all goals have been reached, the agent should stop moving as there is no use for it anymore.
            return "Do Nothing";
        }
    }

    public void pickAction(Agent a){
        String action = getNextAction(a);
        if (action.equals("Update sub-goal")){
            String nextGoal = getNextGoal(a);
            double[] activation = getOutput(getInputSet(nextGoal,a));
            updateDistMap(nextGoal, a, determineOrder(activation));
            setNextGoal(a);
            action = getNextAction(a);
        }
        a.takeAction(action);
        if (model.getAllCells().get(a.getX()).get(a.getY()).isBurning()) {
            removeGoalReached(a);
            backup = a;
            if (debugging) {
                System.out.println("Nr of Agents: " + model.getAgents().size());
            }
        }
        if (use_gui && !action.equals("Do Nothing")) {
            if (showActionFor > 0) {
                sleep(showActionFor);
                showActionFor -= 0;
            }
        }
    }

    public void removeGoalReached(Agent a){
        subGoals.removeGoalReached(a);
    }


    public void setDistance(List<IndexActLink> activationList, String key) {
        int i = 0;
           do {
              subGoals.updateSubGoal(key, activationList.get(i).getIndex());
              i++;
              if (debugging){
                  if (i>7){
                      System.out.println("ACTIVATIONLIST > 7 : " + activationList.size());
                  }
              }
              if (i>=activationList.size()){//TODO: Somehow reset the simulation
//                  resetSimulation("All locations invalid");
              }
            } while (!subGoals.checkSubGoal(key, model.getAgents()));
        if (use_gui && (debugging)) {
            subGoals.paintGoal(key);
        }
    }

    protected void initSubGoalOrder(){ //TODO: If rand float< exploreRate -> make random dist array, otherwise use order of activations
        for (Agent a:model.getAgents()){
            HashMap<String, List<IndexActLink>> activationMap = new HashMap<>();
            for (String s:subGoalKeys){
                double [] outputSet = getOutput(getInputSet(s, a));

                List<IndexActLink> outputList = determineOrder(outputSet);

                activationMap.put(s, outputList);
            }
            subGoalActivation.put(a, activationMap);
        }
        updateDistMap(subGoalActivation);
    }


    private void updateGoalsHit(Agent agent){
        if (agent.isCutting()){
            model.goalsHit++;
            goalToCostMap.get(subGoals.getAgentGoals().get(agent)).setStateXPrime(getInputSet(subGoals.getNextAgentGoal(agent),agent));
        }
        if (debugging){
            System.out.println("# of goals hit: " + model.goalsHit);
        }
    }



    /**
     * Transform state to input vector.
     * @param subGoal: expressed as an integer to allow for use of for loops.
     * @return
     */
//    private double[] getInputSet(int subGoal){
//        float windX = model.getParameters().get("Wind x");
//        float windY = model.getParameters().get("Wind y");
//        double[] set = f.appendArrays(f.cornerVectors(model, false), f.windRelativeToSubgoal(windX, windY, indexMap.get(subGoal)));
//        return set;
//    }

    protected double[] getInputSet(String subGoal, Agent a){
        return f.getInputSet(model, a, subGoal);
    }


    public static double minValue(double[] numbers){
        double min = Double.MAX_VALUE;
        for(int i = 1; i < numbers.length;i++)
        {
            if(numbers[i] < min)
            {
                min = numbers[i];
            }
        }
        return min;
    }

    private void checkAgent(Agent a){
        boolean b = subGoals.getAgentGoals().containsKey(a);
        System.out.println("Agent #" + a.getId() + " of model contained in subGoals list?: " + b);
        for (Agent a2 : subGoals.getAgentGoals().keySet()){
            System.out.println("hashCode agent model: " + a.hashCode() + " -> hashcode agent subGoals: " + a2.hashCode());
        }

    }

    protected void checkAgents(){
        for (Agent a : model.getAgents()) {
            checkAgent(a);
        }
    }

    protected void takeScreenShot(){
        JFrame f = createMainFrame();
        sleep(500);
        screenshot();
        f.dispose();
    }

    protected JFrame createMainFrame(){
        JFrame f = new MainFrame(model);
        sleep(1000);
        return f;
    }

    protected void disposeMainFrame(JFrame f){
        sleep(500);
        f.dispose();
    }

    protected void sleep(int t){
        try {
            Thread.sleep(Math.abs(t));
        } catch (java.lang.InterruptedException e) {
            System.out.println(e.getMessage());
        }
    }

    protected void screenshot(){
        Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
        try {
            BufferedImage capture = new Robot().createScreenCapture(screenRect);
            ImageIO.write(capture, "bmp", new File("./screenshot_run"+ run +"_Algorithm_"+ RLMethod + "_iter_" + iter +".bmp"));

        }catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

    private void writePerformanceFile(){
        String dir = dirGenerator();
        File file = new File(dir);
        if (file.mkdirs() || file.isDirectory()) {
            try {
                FileWriter csvWriter = new FileWriter(dir + "/run_" + run + ".csv");
                csvWriter.append("Iteration");
                csvWriter.append(",");
                csvWriter.append("BurnCost");
                csvWriter.append(",");
                csvWriter.append("MoveCost");
                csvWriter.append(",");
                csvWriter.append("AgentDeathPenalty");
                csvWriter.append("\n");

                for (int i = 0; i< testingIterations; i++){
                    csvWriter.append(i+","+ costArrTesting[i][0]+","+ costArrTesting[i][1]+","+ costArrTesting[i][2]+"\n");
                }

                csvWriter.flush();
                csvWriter.close();
            } catch (IOException e) {
                System.out.println("Some IO-exception occurred");
                e.printStackTrace();
            }
            try {
                BufferedWriter writer = new BufferedWriter(
                        new FileWriter(file + "/run_durations.csv", true)  //Set true for append mode
                );
                writer.write(watch.getTime() + "");
                writer.newLine();   //Add new line
                writer.close();
            } catch (IOException e) {
                System.out.println("Some IO-exception occurred");
                e.printStackTrace();
            }
        } else {
            System.out.println("Unable to make directory");
        }
    }

    protected abstract void resetSimulation();

    protected void resetSubGoals(){
        //SGC = new SubGoalController(algorithm, "CQL", model, rand,  use_gui, debugging);
        assignedGoals = new HashSet<>();
        subGoalActivation = new HashMap<>();
        backup = model.getAgents().get(0);

        double fireLocation[] = f.locationCenterFireAndMinMax(model);

        subGoals = new OrthogonalSubGoals((int)fireLocation[0],(int)fireLocation[1], distMap, algorithm, model.getAllCells());


        initSubGoalOrder();
    }


    public List<IndexActLink> determineOrder(double[] activation){

        List<IndexActLink> outputList = makeIndexActList(activation);

        outputList.sort(Comparator.comparing(IndexActLink::getActivation, Comparator.nullsLast(Comparator.naturalOrder())));

        return outputList;
    }

    protected List<IndexActLink> makeIndexActList(double[] activation){
        List<IndexActLink> outputList = new LinkedList<>();

        for (int i = 0; i<activation.length; i++){
            outputList.add(new IndexActLink(i, (float) activation[i]));
        }

        return outputList;
    }

    private String dirGenerator(){
        return System.getProperty("user.dir") + "/results/" + RLMethod + "/" + algorithm + "/" + model.getNr_agents() + "_agent_environment";
    }

//    private void writePerformanceFile(){
//        String dir = dirGenerator();
//        File file = new File(dir);
//        if (file.mkdirs() || file.isDirectory()) {
//            try {
//                FileWriter csvWriter = new FileWriter(dir + "/run" + run + ".csv");
//                csvWriter.append("Iteration");
//                csvWriter.append(",");
//                csvWriter.append("BurnCost");
//                csvWriter.append(",");
//                csvWriter.append("MoveCost");
//                csvWriter.append(",");
//                csvWriter.append("AgentDeathPenalty");
//                csvWriter.append("\n");
//
//                for (int i = 0; i< testingIterations; i++){
//                    csvWriter.append(i+","+costArrTraining[i][0]+","+costArrTraining[i][1]+","+costArrTraining[i][2]+"\n");
//                }
//
//                csvWriter.flush();
//                csvWriter.close();
//            } catch (IOException e) {
//                System.out.println("Some IO-exception occurred");
//                e.printStackTrace();
//            }
//        } else {
//            System.out.println("Unable to make directory");
//        }
//    }
//
    public void printFinalDistMap(){
        System.out.println("Final distance Map: " + Collections.singletonList(distMap));
        System.out.println("Final number of sub goals assigned: " + goalToCostMap.keySet().size());
        for (Map.Entry<String, InputCost> entry : goalToCostMap.entrySet()) {
            System.out.println(entry.getKey()+" : "+Arrays.toString(entry.getValue().stateX)+" -> "+Arrays.toString(entry.getValue().stateXPrime));
        }
    }

    protected void printSubGoalActivation(){
        for (Agent a:model.getAgents()){
            HashMap<String, List<IndexActLink>> act = subGoalActivation.get(a);
            System.out.println("Sub-goals of agent #" + a.getId());
            for (String s:subGoalKeys){
                System.out.println("\nActivation of sub-goal " + s + ": ");
                for (IndexActLink ial : act.get(s)) {
                    System.out.print( ial.getActivation() + "(" + ial.getIndex()  + ") " );
                }

            }
            System.out.println(" ");
        }
    }

    protected abstract double[] getOutput(double[] input);

    abstract protected double[] getCost();

    protected abstract void train(boolean saveMLP, boolean finalIter);

    protected abstract void test();

    protected abstract void initRL();

    protected abstract String defRLMethod();

    public double getDist(String key){
        return distMap.get(key);
    }

    public String getNextGoal(Agent a){
        return subGoals.getNextAgentGoal(a);
    }

    public void setNextGoal(Agent a){
        subGoals.setNextGoal(a);
    }

    public Map<String, InputCost> getGoalToCostMap() {
        return goalToCostMap;
    }

    //
//    /**
//     * Class needed to order a list containing the index of the distance and the activation of that distance.
//     */
//    public class IndexActLink{
//        private int index;
//        private float activation;
//
//        private IndexActLink(int i, float a){
//            index=i;
//            activation=a;
//        }
//
//        private double getActivation() {
//            return activation;
//        }
//
//        public int getIndex() {
//            return index;
//        }
//    }
//
    public class InputCost{
        private double[] stateX;
        private double[] stateXPrime;
        private double cost;

        private InputCost(){}

        public void setStateX(double[] stateX){
            this.stateX = stateX;
        }

        public void setStateXPrime(double[] stateXPrime) {
            this.stateXPrime = stateXPrime;
        }

        public void setCost(double cost){
            this.cost = cost;
        }

        public double[] getStateX() {
            return stateX;
        }

        public double[] getStateXPrime() {
            return stateXPrime;
        }

        public double getCost() {
            return cost;
        }
    }

    /**
     * Class needed to order a list containing the index of the distance and the activation of that distance.
     */
    public class IndexActLink{
        private int index;
        private float activation;

        public IndexActLink(int i, float a){
            index=i;
            activation=a;
        }

        public double getActivation() {
            return activation;
        }

        public int getIndex() {
            return index;
        }

        public void setIndex(int i){
            index = i;
        }
    }

}
