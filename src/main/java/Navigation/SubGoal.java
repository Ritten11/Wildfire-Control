package Navigation;

import Model.Agent;
import Model.Elements.Element;
import Navigation.PathFinding.BresenhamPath;
import Navigation.PathFinding.DijkstraShortestPath;
import Navigation.PathFinding.PathFinder;

import java.io.Serializable;
import java.util.List;
import java.util.Stack;

public class SubGoal implements Serializable {
    public Element goal;
    List<List<Element>> cells;
    String algorithm;
    public Stack<Element> path;
    public double moveCost;
    private Agent agent;

    public SubGoal(List<List<Element>> cells, Element goal, String algorithm, Agent agent, boolean cutPath){
        this.goal = goal;
        this.cells = cells;
        this.algorithm = algorithm;
        this.agent = agent;
        if (agent.checkTile(goal.getX(), goal.getY())){
            determinePath(cutPath);
        } else {
            System.out.println("Invalid subGoal, pick another one");
        }
    }

    /**
     * If we with to add more path finding methods, we can do so by extending the switch statement.
     */
    private void determinePath(boolean cutPath){
        PathFinder pf;
        switch (algorithm) {
            case "Dijkstra":
                pf = new DijkstraShortestPath(cells, agent, goal, cutPath);
                break;
            default :
                pf = new BresenhamPath(cells, agent, goal, cutPath);
        }
        try {
            pf.findPath();
            path = pf.getPath();
            moveCost = pf.getFinalMoveCost();
        } catch (Exception e){
            System.out.println("Path failed");
            moveCost = Double.MAX_VALUE;
        }
    }

    /**
     * If it is possible to execute the movement necessary to move from the current location to the next location in
     * the path, execute that action. Otherwise do nothing
     * @return
     */

    public String getNextAction() {
        if (path == null){
            System.out.println("Path to subGoal no longer exists");
            return "PathFailed";
        }
        if (path.empty()){
            System.out.println("Path to subGoal is empty");
            return "Do Nothing";
        }
        Element e = path.peek();
        String action = "default";
        int dx = e.getX()-agent.getX();
        int dy = e.getY()-agent.getY();
        if (dx==0){
            if (dy==1){
                action = "Go Up";
            } else if (dy==-1) {
                action = "Go Down";
            } else if (dy==0) {
                //TODO: This is an ad-hoc solution for making the agent dig a path instead of only walking over it.
                // Works for now, should be changed in a more robust function.
                if (cells.get(e.getX()).get(e.getY()).getType().equals("Dirt")){//If the agent is already on dirt, simply skip this step
                    path.pop();
                    return getNextAction();
                }
                action = "Dig";
            }
        } else if (dx==1){
            action = "Go Right";
        } else if (dx==-1){
            action = "Go Left";
        }
        if (agent.tryAction(action)){
            path.pop();
            return action;
        } else {
            return "Do Nothing";
        }
    }

    public boolean pathOnFire(){
        for (Element e : path) {
            if (e.isBurning()) {
                return false;
            }
        }
        return true;
    }

    public boolean pathExists() { return !(path==null||path.empty()); }

    /**
     * debugging function for checking the optimal path
     */
    public void printPath(Stack<Element> path) {
        System.out.println("shortest path found from subGoal "+ goal.toCoordinates() +":");
        for (Element e:path){
            System.out.println("-> (" + e.getX() + ", " + e.getY() + ")");
        }
        System.out.println("Agent at: (" + agent.getX() + ", " + agent.getY() + ")");
    }

    public Stack<Element> getPath() {
        return path;
    }

    public double getMoveCost() {
        return moveCost;
    }
}
