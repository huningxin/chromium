// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

suite('<bookmarks-command-manager>', function() {
  let commandManager;
  let store;
  let lastCommand;
  let lastCommandIds;
  let bmpCopyFunction;
  let bmpPasteFunction;

  suiteSetup(function() {
    // Overwrite bookmarkManagerPrivate APIs which will crash if called with
    // fake data.
    bmpCopyFunction = chrome.bookmarkManagerPrivate.copy;
    bmpPasteFunction = chrome.bookmarkManagerPrivate.paste;
    chrome.bookmarkManagerPrivate.copy = function() {};
    chrome.bookmarkManagerPrivate.removeTrees = function() {};
  });

  suiteTeardown(function() {
    chrome.bookmarkManagerPrivate.copy = bmpCopyFunction;
    chrome.bookmarkManagerPrivate.paste = bmpPasteFunction;
  });

  setup(function() {
    const bulkChildren = [];
    for (let i = 1; i <= 20; i++) {
      const id = '3' + i;
      bulkChildren.push(createItem(id, {url: `http://${id}/`}));
    }

    store = new bookmarks.TestStore({
      nodes: testTree(
          createFolder(
              '1',
              [
                createFolder(
                    '11',
                    [
                      createItem('111', {url: 'http://111/'}),
                    ]),
                createFolder(
                    '12',
                    [
                      createItem('121', {url: 'http://121/'}),
                      createFolder(
                          '122',
                          [
                            createItem('1221'),
                          ]),
                    ]),
                createItem('13', {url: 'http://13/'}),
              ]),
          createFolder(
              '2',
              [
                createFolder('21', []),
              ]),
          createFolder('3', bulkChildren),
          createFolder('4', [], {unmodifiable: 'managed'})),
      selectedFolder: '1',
    });
    store.replaceSingleton();

    commandManager = new TestCommandManager();
    replaceBody(commandManager);
    document.body.appendChild(
        document.createElement('bookmarks-toast-manager'));
    bookmarks.DialogFocusManager.instance_ = null;
  });

  test('Copy URL is only active for single URL items', function() {
    assertFalse(commandManager.canExecute(Command.COPY_URL, new Set(['11'])));
    assertFalse(
        commandManager.canExecute(Command.COPY_URL, new Set(['11', '13'])));
    assertTrue(commandManager.canExecute(Command.COPY_URL, new Set(['13'])));
  });

  test('context menu hides invalid commands', function() {
    store.data.selection.items = new Set(['11', '13']);
    store.notifyObservers();

    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.ITEM);
    Polymer.dom.flush();

    const commandHidden = {};
    commandManager.root.querySelectorAll('.dropdown-item').forEach(element => {
      commandHidden[element.getAttribute('command')] = element.hidden;
    });

    // With a folder and an item selected, the only available context menu item
    // is 'Delete'.
    assertTrue(commandHidden[Command.EDIT]);
    assertTrue(commandHidden[Command.COPY_URL]);
    assertFalse(commandHidden[Command.DELETE]);
  });

  test('edit shortcut triggers when valid', function() {
    const key = cr.isMac ? 'Enter' : 'F2';

    store.data.selection.items = new Set(['13']);
    store.notifyObservers();

    MockInteractions.pressAndReleaseKeyOn(document.body, '', [], key);
    commandManager.assertLastCommand(Command.EDIT, ['13']);

    // Doesn't trigger when multiple items are selected.
    store.data.selection.items = new Set(['11', '13']);
    store.notifyObservers();

    MockInteractions.pressAndReleaseKeyOn(document.body, '', [], key);
    commandManager.assertLastCommand(null);

    // Doesn't trigger when nothing is selected.
    store.data.selection.items = new Set();
    store.notifyObservers();

    MockInteractions.pressAndReleaseKeyOn(document.body, '', [], key);
    commandManager.assertLastCommand(null);
  });

  test('delete command triggers', function() {
    store.data.selection.items = new Set(['12', '13']);
    store.notifyObservers();

    MockInteractions.pressAndReleaseKeyOn(document.body, 46, '', 'Delete');
    commandManager.assertLastCommand(Command.DELETE, ['12', '13']);
  });

  test('copy command triggers', function() {
    const modifier = cr.isMac ? 'meta' : 'ctrl';

    store.data.selection.items = new Set(['11', '13']);
    store.notifyObservers();

    MockInteractions.pressAndReleaseKeyOn(document.body, '', modifier, 'c');
    commandManager.assertLastCommand(Command.COPY, ['11', '13']);
  });

  test('cut/paste commands trigger', function() {
    let lastCut;
    let lastPaste;
    chrome.bookmarkManagerPrivate.cut = function(idList) {
      lastCut = idList.sort();
    };
    chrome.bookmarkManagerPrivate.paste = function(selectedFolder) {
      lastPaste = selectedFolder;
    };

    store.data.selection.items = new Set(['11', '13']);
    store.notifyObservers();

    const modifier = cr.isMac ? 'meta' : 'ctrl';
    MockInteractions.pressAndReleaseKeyOn(document.body, '', modifier, 'x');
    assertDeepEquals(['11', '13'], lastCut);
    MockInteractions.pressAndReleaseKeyOn(document.body, '', modifier, 'v');
    assertEquals('1', lastPaste);
  });

  test('undo and redo commands trigger', function() {
    const undoModifier = cr.isMac ? 'meta' : 'ctrl';
    const undoKey = 'z';
    const redoModifier = cr.isMac ? ['meta', 'shift'] : 'ctrl';
    const redoKey = cr.isMac ? 'Z' : 'y';

    MockInteractions.pressAndReleaseKeyOn(
        document.body, '', undoModifier, undoKey);
    commandManager.assertLastCommand(Command.UNDO);

    MockInteractions.pressAndReleaseKeyOn(
        document.body, '', redoModifier, redoKey);
    commandManager.assertLastCommand(Command.REDO);
  });

  test('Show In Folder is only available during search', function() {
    store.data.selection.items = new Set(['12']);
    store.notifyObservers();

    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.ITEM);
    Polymer.dom.flush();

    const showInFolderItem = commandManager.root.querySelector(
        `[command='${Command.SHOW_IN_FOLDER}']`);

    // Show in folder hidden when search is inactive.
    assertTrue(showInFolderItem.hidden);

    // Show in Folder visible when search is active.
    store.data.search.term = 'test';
    store.data.search.results = ['12', '13'];
    store.notifyObservers();
    commandManager.closeCommandMenu();
    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.ITEM);
    assertFalse(showInFolderItem.hidden);

    // Show in Folder hidden when menu is opened from the sidebar.
    commandManager.closeCommandMenu();
    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.TREE);
    assertTrue(showInFolderItem.hidden);

    // Show in Folder hidden when multiple items are selected.
    store.data.selection.items = new Set(['12', '13']);
    store.notifyObservers();
    commandManager.closeCommandMenu();
    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.ITEM);
    assertTrue(showInFolderItem.hidden);

    // Executing the command selects the parent folder.
    commandManager.handle(Command.SHOW_IN_FOLDER, new Set(['12']));
    assertEquals('select-folder', store.lastAction.name);
    assertEquals('1', store.lastAction.id);
  });

  test('does not delete children at same time as ancestor', function() {
    let lastDelete = null;
    chrome.bookmarkManagerPrivate.removeTrees = function(idArray) {
      lastDelete = idArray.sort();
    };

    const parentAndChildren = new Set(['11', '12', '111', '1221']);
    assertTrue(commandManager.canExecute(Command.DELETE, parentAndChildren));
    commandManager.handle(Command.DELETE, parentAndChildren);

    assertDeepEquals(['11', '12'], lastDelete);
  });

  test('expandUrls_ expands one level of URLs', function() {
    let urls = commandManager.expandUrls_(new Set(['1']));
    assertDeepEquals(['http://13/'], urls);

    urls = commandManager.expandUrls_(new Set(['11', '12', '13']));
    assertDeepEquals(['http://111/', 'http://121/', 'http://13/'], urls);
  });

  test('shift-enter opens URLs in new window', function() {
    store.data.selection.items = new Set(['12', '13']);
    store.notifyObservers();

    let lastCreate;
    chrome.windows.create = function(createConfig) {
      lastCreate = createConfig;
    };

    MockInteractions.pressAndReleaseKeyOn(document.body, 13, 'shift', 'Enter');
    commandManager.assertLastCommand(Command.OPEN_NEW_WINDOW, ['12', '13']);
    assertDeepEquals(['http://121/', 'http://13/'], lastCreate.url);
    assertFalse(lastCreate.incognito);
  });

  test('shift-enter does not trigger enter commands', function() {
    // Enter by itself performs an edit (Mac) or open (non-Mac). Ensure that
    // shift-enter doesn't trigger those commands.
    store.data.selection.items = new Set(['13']);
    store.notifyObservers();

    MockInteractions.pressAndReleaseKeyOn(document.body, 13, 'shift', 'Enter');
    commandManager.assertLastCommand(Command.OPEN_NEW_WINDOW);
  });

  test('opening many items causes a confirmation dialog', function() {
    let lastCreate = null;
    chrome.windows.create = function(createConfig) {
      lastCreate = createConfig;
    };

    const items = new Set(['3']);
    assertTrue(commandManager.canExecute(Command.OPEN_NEW_WINDOW, items));

    commandManager.handle(Command.OPEN_NEW_WINDOW, items);
    // No window should be created right away.
    assertEquals(null, lastCreate);

    const dialog = commandManager.$.openDialog.getIfExists();
    assertTrue(dialog.open);

    // Pressing 'cancel' should not open the window.
    MockInteractions.tap(dialog.querySelector('.cancel-button'));
    assertFalse(dialog.open);
    assertEquals(null, lastCreate);

    commandManager.handle(Command.OPEN_NEW_WINDOW, items);
    assertTrue(dialog.open);

    // Pressing 'yes' will open all the URLs.
    MockInteractions.tap(dialog.querySelector('.action-button'));
    assertFalse(dialog.open);
    assertEquals(20, lastCreate.url.length);
  });

  test('cannot execute "Open in New Tab" on folders with no items', function() {
    const items = new Set(['2']);
    assertFalse(commandManager.canExecute(Command.OPEN_NEW_TAB, items));

    store.data.selection.items = items;

    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.ITEM);
    Polymer.dom.flush();

    const commandItem = {};
    commandManager.root.querySelectorAll('.dropdown-item').forEach(element => {
      commandItem[element.getAttribute('command')] = element;
    });

    assertTrue(commandItem[Command.OPEN_NEW_TAB].disabled);
    assertFalse(commandItem[Command.OPEN_NEW_TAB].hidden);

    assertTrue(commandItem[Command.OPEN_NEW_WINDOW].disabled);
    assertFalse(commandItem[Command.OPEN_NEW_WINDOW].hidden);

    assertTrue(commandItem[Command.OPEN_INCOGNITO].disabled);
    assertFalse(commandItem[Command.OPEN_INCOGNITO].hidden);
  });

  test('cannot execute editing commands when editing is disabled', function() {
    const items = new Set(['12']);

    store.data.prefs.canEdit = false;
    store.data.selection.items = items;
    store.notifyObservers();

    assertFalse(commandManager.canExecute(Command.EDIT, items));
    assertFalse(commandManager.canExecute(Command.DELETE, items));
    assertFalse(commandManager.canExecute(Command.UNDO, items));
    assertFalse(commandManager.canExecute(Command.REDO, items));

    // No divider line should be visible when only 'Open' commands are enabled.
    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.ITEM);
    commandManager.root.querySelectorAll('hr').forEach(element => {
      assertTrue(element.hidden);
    });
  });

  test('cannot edit unmodifiable nodes', function() {
    // Cannot edit root folders.
    let items = new Set(['1']);
    store.data.selection.items = items;
    assertFalse(commandManager.canExecute(Command.EDIT, items));
    assertFalse(commandManager.canExecute(Command.DELETE, items));

    items = new Set(['4']);
    assertFalse(commandManager.canExecute(Command.EDIT, items));
    assertFalse(commandManager.canExecute(Command.DELETE, items));

    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.ITEM);
    const commandItem = {};
    commandManager.root.querySelectorAll('.dropdown-item').forEach(element => {
      commandItem[element.getAttribute('command')] = element;
    });
    MockInteractions.tap(commandItem[Command.EDIT]);
    commandManager.assertLastCommand(null);
  });

  test('keyboard shortcuts are disabled while a dialog is open', function() {
    assertFalse(bookmarks.DialogFocusManager.getInstance().hasOpenDialog());
    items = new Set(['12']);
    store.data.selection.items = items;
    store.notifyObservers();

    const editKey = cr.isMac ? 'Enter' : 'F2';
    MockInteractions.pressAndReleaseKeyOn(document.body, '', '', editKey);
    commandManager.assertLastCommand(Command.EDIT);
    assertTrue(bookmarks.DialogFocusManager.getInstance().hasOpenDialog());

    MockInteractions.pressAndReleaseKeyOn(document.body, '', '', 'Delete');
    commandManager.assertLastCommand(null);
  });

  test('toolbar menu options are disabled when appropriate', function() {
    store.data.selectedFolder = '1';
    store.data.prefs.canEdit = true;
    store.notifyObservers();

    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.TOOLBAR);
    assertTrue(commandManager.canExecute(Command.SORT, new Set()));
    assertTrue(commandManager.canExecute(Command.ADD_BOOKMARK, new Set()));
    assertTrue(commandManager.canExecute(Command.ADD_FOLDER, new Set()));

    store.data.selectedFolder = '4';
    store.notifyObservers();

    assertFalse(commandManager.canExecute(Command.SORT, new Set()));
    assertFalse(commandManager.canExecute(Command.ADD_BOOKMARK, new Set()));
    assertFalse(commandManager.canExecute(Command.ADD_FOLDER, new Set()));
    assertTrue(commandManager.canExecute(Command.IMPORT, new Set()));

    store.data.selectedFolder = '1';
    store.data.prefs.canEdit = false;
    store.notifyObservers();

    assertFalse(commandManager.canExecute(Command.SORT, new Set()));
    assertFalse(commandManager.canExecute(Command.IMPORT, new Set()));
    assertFalse(commandManager.canExecute(Command.ADD_BOOKMARK, new Set()));
    assertFalse(commandManager.canExecute(Command.ADD_FOLDER, new Set()));
  });

  test('sort button is disabled when folder is empty', function() {
    store.data.selectedFolder = '3';
    store.notifyObservers();

    commandManager.openCommandMenuAtPosition(0, 0, MenuSource.TOOLBAR);
    assertTrue(commandManager.canExecute(Command.SORT, new Set()));

    store.data.selectedFolder = '21';
    store.notifyObservers();

    assertFalse(commandManager.canExecute(Command.SORT, new Set()));

    // Adding 2 bookmarks should enable sorting.
    store.setReducersEnabled(true);
    const item1 = {
      id: '211',
      parentId: '21',
      index: 0,
      url: 'https://www.example.com',
    };
    store.dispatch(bookmarks.actions.createBookmark(item1.id, item1));
    assertFalse(commandManager.canExecute(Command.SORT, new Set()));

    const item2 = {
      id: '212',
      parentId: '21',
      index: 1,
      url: 'https://www.example.com',
    };
    store.dispatch(bookmarks.actions.createBookmark(item2.id, item2));
    assertTrue(commandManager.canExecute(Command.SORT, new Set()));
  });
});

suite('<bookmarks-item> CommandManager integration', function() {
  let list;
  let items;
  let commandManager;
  let openedTabs;

  setup(function() {
    store = new bookmarks.TestStore({
      nodes: testTree(createFolder(
          '1',
          [
            createFolder(
                '11',
                [
                  createItem('111', {url: 'http://111/'}),
                ]),
            createItem('12', {url: 'http://12/'}),
            createItem('13', {url: 'http://13/'}),
          ])),
      selectedFolder: '1',
    });
    store.setReducersEnabled(true);
    store.replaceSingleton();

    commandManager = document.createElement('bookmarks-command-manager');

    list = document.createElement('bookmarks-list');
    replaceBody(list);
    document.body.appendChild(commandManager);
    Polymer.dom.flush();

    items = list.root.querySelectorAll('bookmarks-item');

    openedTabs = [];
    chrome.tabs.create = function(createConfig) {
      openedTabs.push(createConfig);
    };
  });

  function assertOpenedTabs(tabs) {
    assertDeepEquals(tabs, openedTabs.map(createConfig => createConfig.url));
  }

  function simulateDoubleClick(element, config) {
    config = config || {};
    customClick(element, config);
    config.detail = 2;
    customClick(element, config);
  }

  function simulateMiddleClick(element, config) {
    config = config || {};
    config.button = 1;
    customClick(element, config, 'auxclick');
  }

  test('double click opens folders in bookmark manager', function() {
    simulateDoubleClick(items[0]);
    assertEquals(store.data.selectedFolder, '11');
  });

  test('double click opens items in foreground tab', function() {
    simulateDoubleClick(items[1]);
    assertOpenedTabs(['http://12/']);
  });

  test('shift-double click opens full selection', function() {
    // Shift-double click works because the first click event selects the range
    // of items, then the second doubleclick event opens that whole selection.
    customClick(items[0]);
    simulateDoubleClick(items[1], {shiftKey: true});

    assertOpenedTabs(['http://111/', 'http://12/']);
  });

  test('control-double click opens full selection', function() {
    customClick(items[0]);
    simulateDoubleClick(items[2], {ctrlKey: true});

    assertOpenedTabs(['http://111/', 'http://13/']);
  });

  test('middle-click opens clicked item in new tab', function() {
    // Select multiple items.
    customClick(items[1]);
    customClick(items[2], {shiftKey: true});

    // Only the middle-clicked item is opened.
    simulateMiddleClick(items[2]);
    assertDeepEquals(['13'], normalizeIterable(store.data.selection.items));
    assertOpenedTabs(['http://13/']);
    assertFalse(openedTabs[0].active);
  });

  test('middle-click does not open folders', function() {
    simulateMiddleClick(items[0]);
    assertDeepEquals(['11'], normalizeIterable(store.data.selection.items));
    assertOpenedTabs([]);
  });

  test('shift-middle click opens in foreground tab', function() {
    simulateMiddleClick(items[1], {shiftKey: true});
    assertOpenedTabs(['http://12/']);
    assertTrue(openedTabs[0].active);
  });
});

suite('<bookmarks-command-manager> whole page integration', function() {
  let app;
  let store;
  let commandManager;

  let testFolderId;

  function create(bookmark) {
    return new Promise(function(resolve) {
      chrome.bookmarks.create(bookmark, resolve);
    });
  }

  suiteSetup(function() {
    const testFolder = {
      parentId: '1',
      title: 'Test',
    };
    return create(testFolder).then(function(testFolderNode) {
      testFolderId = testFolderNode.id;
      const testItem = {
        parentId: testFolderId,
        title: 'Test bookmark',
        url: 'https://www.example.com/',
      };
      return Promise.all([
        create(testItem),
        create(testItem),
      ]);
    });
  });

  setup(function() {
    store = new bookmarks.TestStore({});
    store.replaceSingleton();
    store.setReducersEnabled(true);
    const promise = store.acceptInitOnce();
    const app = document.createElement('bookmarks-app');
    replaceBody(app);

    commandManager = bookmarks.CommandManager.getInstance();

    return promise.then(() => {
      store.dispatch(bookmarks.actions.selectFolder(testFolderId));
    });
  });

  test('paste selects newly created items', function() {
    const displayedIdsBefore = bookmarks.util.getDisplayedList(store.data);
    commandManager.handle(Command.SELECT_ALL, new Set());
    commandManager.handle(Command.COPY, new Set(displayedIdsBefore));

    store.expectAction('select-items');
    commandManager.handle(Command.PASTE, new Set());

    return store.waitForAction('select-items').then(function(action) {
      const displayedIdsAfter = bookmarks.util.getDisplayedList(store.data);
      assertEquals(4, displayedIdsAfter.length);

      // The start of the list shouldn't change.
      assertEquals(displayedIdsBefore[0], displayedIdsAfter[0]);
      assertEquals(displayedIdsBefore[1], displayedIdsAfter[1]);

      // The two pasted items should be selected at the end of the list.
      assertEquals(action.items[0], displayedIdsAfter[2]);
      assertEquals(action.items[1], displayedIdsAfter[3]);
      assertEquals(2, action.items.length);
      assertEquals(action.anchor, displayedIdsAfter[2]);
    });
  });

  suiteTeardown(function(done) {
    chrome.bookmarks.removeTree(testFolderId, () => done());
  });
});
