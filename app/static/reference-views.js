const referenceLimit = 6;
const referenceBytes = 24 * 1024 ** 2;
const referenceTypes = new Set(['image/png', 'image/jpeg', 'image/webp']);

function validateReferences(files) {
    if (files.length > referenceLimit || files.reduce((bytes, file) => bytes + file.size, 0) > referenceBytes) {
        return 'Choose up to six images, totaling at most 24 MiB.';
    }
    if (files.some(file => !file.size || !referenceTypes.has(file.type))) {
        return 'Choose nonempty PNG, JPG or WebP images.';
    }
    return '';
}

const referenceDrafts = {
    edit(action) {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('meshport-view-handoff', 1);
            request.onupgradeneeded = () => {
                request.result.createObjectStore('drafts').createIndex('expiry', 'expiry');
            };
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                const db = request.result;
                const transaction = db.transaction('drafts', 'readwrite');
                let result;
                transaction.oncomplete = () => {
                    db.close();
                    resolve(result);
                };
                transaction.onabort = () => {
                    db.close();
                    reject(transaction.error);
                };
                try {
                    action(transaction.objectStore('drafts'), value => { result = value; });
                } catch (error) {
                    transaction.abort();
                    reject(error);
                }
            };
        });
    },

    stage(files) {
        const error = validateReferences(files.map(item => item.file || item));
        if (!files.length || error) return Promise.reject(new Error(error || 'Select at least one crop.'));
        const token = crypto.randomUUID();
        const now = Date.now();
        return this.edit((store, result) => {
            const expired = store.index('expiry').openKeyCursor(IDBKeyRange.upperBound(now));
            expired.onsuccess = () => {
                const cursor = expired.result;
                if (!cursor) return;
                store.delete(cursor.primaryKey);
                cursor.continue();
            };
            store.add({ files, expiry: now + 10 * 60 * 1000 }, token);
            result(token);
        });
    },

    take(token) {
        if (!/^[a-f0-9-]{36}$/.test(token)) return Promise.resolve(null);
        return this.edit((store, result) => {
            const request = store.get(token);
            request.onsuccess = () => {
                const draft = request.result;
                if (!draft) return;
                store.delete(token);
                if (draft.expiry > Date.now()) result(draft.files);
            };
        });
    },
};
