# Discriminating Cosmological Scenarios By 3DCNN
Weak gravitational lensing is one of the most promising cosmological probes of the late universe. Several large ongoing (DES, KiDS, HSC) and planned (LSST, Euclid, WFIRST) astronomical surveys attempt to collect even deeper and larger scale data on weak lensing. Due to gravitational collapse, the distribution of dark matter is non-Gaussian on small scales. Previous studies attempted to extract non-Gaussian information from weak lensing observations through several higher order statistics such as the three-point correlation function, peak counts.  We demonstrate that a CNN is able to yield significantly
```python
model.add(Conv3D(8, (2, 3, 10), input_shape=(1, 4, 5, 100),
                         padding="same", activation='relu'))
        model.add(Conv3D(8, (2, 3, 10), padding="same", activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 1, 5)))
        model.add(Conv3D(8, (2, 3, 10), padding="same", activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 1, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        optimizer = Adam(lr=0.001, decay=0.001) 
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
```                     
                      
