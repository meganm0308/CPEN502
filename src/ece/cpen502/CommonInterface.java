package ece.cpen502;

import java.io.File;
import java.io.IOException;
/**
 * This interface is common to both the Neural Net and LUT interfaces.
 * The idea is that you should be able to easily switch the LUT
 * for the Neural Net since the interfaces are identical.
 *
 * @author sarbjit
 * @date 20 June 2012
 */
public interface CommonInterface {
    /**
     * A method to write either a LUT or weights of an neural net to a file.
     *
     * @param filePath of type String.
     */
    public void save(String filePath);

    /**
     * Loads the LUT or neural net weights from file. The load must of course
     * have knowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being
     * made to load data into an LUT or neural net whose structure does not match
     * the data in the file. (e.g. wrong number of hidden neurons).
     *
     * @throws IOException
     */
    public void load(String argFileName) throws IOException;
}
