def __getitem__(self, idx):
        images = []
        measurments = []
        for b in range(self.batch_size):
            images.append([])

        for i in range(len(self.input_measures)):
            measurments.append([])
            for b in range(self.batch_size):
                measurments[i].append([])
        y = []
        for i in range(len(self.output_measures)):
            y.append([])

        cur_idx = idx*self.batch_size
        for b in range(self.batch_size):
            #print("new_batch")
            # Add the current sequence to the batch
            sequence = self.data[cur_idx]
            for j in range(self.seq_len):
                #print("new_row")
                row = sequence.iloc[j, :]
                #print(row.frame)
                images[b].append(self.get_image(row))
                input_measurements = row[self.input_measures]
                for i, measure in enumerate(self.input_measures):
                    if measure == "Speed":
                        measurments[i][b].append([input_measurements[measure]])
                    else:
                        measurments[i][b].append(input_measurements[measure])
            
            row = sequence.iloc[-1, :]
            output_measurements = row[self.output_measures]
            for i, measure in enumerate(self.output_measures):
                y[i].append(output_measurements[measure])
            cur_idx += 1

        # Convert x to dict to allow for multiple inputs
        X = {}
        X["input_Image"] = np.array(images)
        for i, measure in enumerate(self.input_measures):
            X["input_" + measure] = np.array(measurments[i])
        Y = {}
        for i, measure in enumerate(self.output_measures):
            Y["output_" + measure] = np.array(y[i])
        return X, Y
